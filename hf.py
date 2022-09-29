"""
huggingface/mlperf inference benchmarking tool

example:
python hf.py --model bert-base-uncased --task questionanswering --dataset squad --scenario Server --find-peak-performance
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import logging
import os
import re
import sys
import threading
import time
from queue import Queue

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
import onnxruntime as ort
from datasets import load_dataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring


SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

TASKS = ["pretraining", "lm", "base", "causualm", "masklm", "seq2seqlm", "classification", "multiplechoise", "nextsentence",
         "tokenclassification", "questionanswering", "tablequestionanswering", "imageclassification", "vision2seq",
         "visualquestionanswering", "audioclassification", "audioframeclassification", "ctc", "speechseq2seq",
         "audioxvector", "maskimagemodeling", "objectdetection", "imagesegmentation", "semanticsegmentation",
         "instancesegmentation", "summary", "translate"]

TASK2DS = {"lm": "wikitext", "masklm": "wikitext", "causualm": "wikitext", "questionanswering": "squad"}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="pytorch", help="backend override, one of ")
    parser.add_argument("--model", required=True, help="huggingface model")
    parser.add_argument("--task", help="task type")
    parser.add_argument("--dataset", help="huggingface dataset")
    parser.add_argument("--scenario", default="SingleStream", choices=list(SCENARIO_MAP.keys()), help="mlperf benchmark scenario, one of ")
    parser.add_argument("--model-task", default="default", help="model from profile to run")
    parser.add_argument("--name", required=True, help="name of the run")
    parser.add_argument("--cache-dir", default="/tmp", help="cache directory")
    parser.add_argument("--output", default="results", help="test results")
    parser.add_argument("--csv", help="csv output")
    parser.add_argument("--threads", default=2, type=int, help="threads")
    parser.add_argument("--batchsize", type=int, help="batchsize")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--fake", type=int, help="use fake data with seqlen")
    parser.add_argument("--profile", action="store_true", help="profile")
    parser.add_argument("--debug", action="store_true", help="debug, turn traces on")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="precision")

    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf-config", help="extra mlperf rules")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--queries", help="min_query_count")
    parser.add_argument("--max-latency-percentile", type=float, help="mlperf max latency pct tile")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", type=int, help="mlperf sample per query")
    args = parser.parse_args()

    cols = args.model.split(":")
    if len(cols) > 1:
        args.task = cols[1]
        args.model = cols[0]
    if args.task not in TASKS:
        parser.error(f"--task must be one of {TASKS}")
    if not args.model_task:
        args.model_task = args.task
    if not args.dataset:
        args.dataset = TASK2DS.get(args.task)
    if args.dataset not in ["squad", "wikitext"]:
        parser.error("--dataset must be one of squad,wikitext")
    if args.accuracy and args.fake:
        parser.error("--accuracy and --fake don't work together")
    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    if args.scenario == "Server" and not args.qps and not args.find_peak_performance:
        parser.error("Server scenario requires --qps")

    if args.output:
        args.output = os.path.abspath(args.output)

    return args


class PostProcessNone:
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, results, ids):
        processed_results = []
        for idx in ids:
            processed_results.append([0])
        return processed_results


class Backend():
    def __init__(self, max_input_size, kwargs):
        self.max_input_size = max_input_size
        self.generative = kwargs.get("generative")
        self.cuda = False
        self.parameters = 0
        self.dummy_inputs = []
        self.input_mapping = None

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def tensor_format(self):
        raise NotImplementedError("Backend:tensor_format")

    def load(self, model_path, inputs=None, outputs=None, profile=False):
        raise NotImplementedError("Backend:load")

    def predict(self, feed):
        raise NotImplementedError("Backend:predict")


class BackendOnnxruntime(Backend):
    def __init__(self, model, tokenizer, model_name, model_class, max_input_size, kwargs):
        super().__init__(max_input_size, kwargs)
        print(f"Parameters: {model.num_parameters() / 1000000:.1f}M")
        self.parameters = model.num_parameters()
        self.dummy_inputs = model.dummy_inputs
        providers = ['CPUExecutionProvider']
        if ort.get_device() == "GPU":
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus is None or len(gpus) > 0:
                providers = ['CUDAExecutionProvider']
                self.cuda = True

        log.info("onnxruntime providers: %s", providers)

        # export pytorch to onnx
        from onnxruntime.transformers.onnx_exporter import export_onnx_model_from_pt
        from onnxruntime.transformers.benchmark_helper import ConfigModifier
        from onnxruntime.transformers.benchmark import parse_arguments
        from onnxruntime.transformers.huggingface_models import MODEL_CLASSES, MODELS

        # hacking around some cross module enum issue
        sys.argv = ["--model", "--precision", kwargs.get("precision")]
        fake_args = parse_arguments()
        precision = fake_args.precision
        optimizer_info = fake_args.optimizer_info

        # model_name = model_name.replace("/", "_")
        model_inputs = tokenizer.model_input_names
        opset = 14
        use_external_data_format = False
        cache_dir = None
        onnx_dir = "/tmp/onnx"
        use_raw_attention_mask = True
        model_fusion_statistics = {}
        fusion_options = None
        use_gpu = self.cuda
        layers = None
        try:
            layers = model.config.n_layers
        except:
            layers = model.config.num_hidden_layers

        if model_name.startswith("t5-"):
            from onnxruntime.transformers.models.t5.convert_to_onnx import export_onnx_models
            use_decoder_start_token = False
            separate_encoder_and_decoder_init = False
            onnx_model_file = export_onnx_models(model_name, cache_dir, onnx_dir, use_gpu, use_external_data_format, True,
                                                 kwargs.get("precision"), False, use_decoder_start_token,
                                                 not separate_encoder_and_decoder_init, False, False, False)
            onnx_model_file = onnx_model_file[0]
            self.input_mapping = {"input_ids": "encoder_input_ids", "attention_mask": "encoder_attention_mask", "decoder_input_ids": "decoder_input_ids"}
        else:
            model_type = MODELS[model_name][3]
            config = ConfigModifier(layers)
            tokenizer = None
            model = None

            with torch.no_grad():
                onnx_model_file, is_valid_onnx_model, vocab_size, max_sequence_length = \
                    export_onnx_model_from_pt(model_name, opset, use_external_data_format, model_type, model_class, config, cache_dir,
                                              onnx_dir, model_inputs, use_gpu, precision, optimizer_info, False, use_raw_attention_mask,
                                              False, model_fusion_statistics, fusion_options)

        opt = ort.SessionOptions()
        self.sess = ort.InferenceSession(onnx_model_file, opt, providers=providers)

    def version(self):
        return ort.__version__

    def name(self):
        return "onnxruntime"

    def tensor_format(self):
        return "np"

    def predict(self, feed):
        # print(feed)
        if self.generative:
            ret = self.model.generate(**feed, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
        else:
            ret = self.sess.run([], feed)
        return ret


class BackendPytorch(Backend):
    def __init__(self, model, tokenizer, max_input_size, kwargs):
        super().__init__(max_input_size, kwargs)
        print(f"Parameters: {model.num_parameters() / 1000000:.1f}M")
        self.parameters = model.num_parameters()
        self.dummy_inputs = model.dummy_inputs
        if torch.cuda.is_available():
            model = model.cuda()
            self.cuda = True
        if kwargs.get("precision") == "fp16":
            model.half()
        self.model = model
        if kwargs.get("jit"):
            self.model = torch.jit.trace(model, input_ids)
        torch.set_grad_enabled(False)
        log.info(f"tokenizer inputs={tokenizer.model_input_names}")
        log.info(f"model inputs={self.dummy_inputs}")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch"

    def tensor_format(self):
        return "pt"

    def predict(self, feed):
        # print(feed)
        if self.generative:
            ret = self.model.generate(**feed, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
        else:
            ret = self.model(**feed)
        return ret


class BackendDeepspeed(Backend):

    def __init__(self, model, max_input_size, kwargs):
        super().__init__(max_input_size, kwargs)
        import deepspeed

        dtype = torch.half if kwargs.get("precision") == "fp16" else torch.float

        if torch.cuda.is_available():
            model = model.cuda()
            self.cuda = True

        self.ds_engine = deepspeed.init_inference(model, dtype=dtype, checkpoint=None, replace_method='auto', replace_with_kernel_inject=True)
        model = model = self.ds_engine.module

        print(f"Parameters: {model.num_parameters() / 1000000:.1f}M")
        self.model = model
        self.dummy_inputs = model.dummy_inputs
        torch.set_grad_enabled(False)

    def version(self):
        return torch.__version__

    def name(self):
        return "deepspeed"

    def tensor_format(self):
        return "pt"

    def predict(self, feed):
        if self.generative:
            ret = self.model.generate(**feed, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
        else:
            ret = self.model(**feed)
        return ret


class RunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.max_batchsize = max_batchsize
        self.errors = 0

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, take_accuracy):
        pass

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        query_id, content_id, feed = qitem
        try:
            results = self.model.predict(feed)
            processed_results = self.post_process(results, content_id)
        except Exception as ex:  # pylint: disable=broad-except
            # src = [self.ds.get_item_loc(i) for i in content_id]
            src = ""
            self.errors += 1
            if self.errors > 10:
                log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, qid in enumerate(query_id):
                response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            feed = self.ds.make_batch(self.model, idx)
            self.run_one_item((query_id, idx, feed))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                feed = self.ds.make_batch(self.model, idx[i:i + bs])
                self.run_one_item((query_id[i:i + bs], idx[i:i + bs], feed))

    def finish(self):
        pass


class QueueRunner(RunnerBase):
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        super().__init__(model, ds, threads, post_proc, max_batchsize)
        self.tasks = Queue(maxsize=threads * 4)
        self.workers = []

        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            tasks_queue.task_done()
            if not qitem:
                # None in the queue indicates the parent want us to exit
                break
            self.run_one_item(qitem)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            # queue batch
            feed = self.ds.make_batch(self.model, idx)
            self.tasks.put((query_id, idx, feed))
        else:
            # batch is to large, split into multiple batches
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                ie = i + bs
                feed = self.ds.make_batch(self.model, idx[i:ie])
                self.tasks.put((query_id[i:ie], idx[i:ie], feed))

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()


class BatchingQueueRunner(RunnerBase):
    """Runner that implements dynamic batching."""

    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        super().__init__(model, ds, threads, post_proc, max_batchsize)
        self.cv = threading.Condition()
        self.done = False
        self.query_samples = []
        self.workers = []
        self.threads = threads

        # start all threads
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.cv,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()
        time.sleep(1)

    def handle_tasks(self, cv):
        """Worker thread."""
        max_batchsize = self.max_batchsize
        stats = [0] * (max_batchsize + 1)
        while True:
            with cv:
                # wait for something to do
                while len(self.query_samples) == 0 and not self.done:
                    cv.wait()
                query_samples = self.query_samples
                if len(query_samples) > max_batchsize:
                    # only take max_batchsize
                    self.query_samples = query_samples[max_batchsize:]
                    # wake up somebody to take care of it
                    cv.notify(n=1)
                else:
                    # swap the entire queue
                    self.query_samples = []
            if self.done:
                # parent wants us to exit
                print("count per batchsize:", stats)
                break
            # run inference, lock is released
            query_samples = query_samples[:max_batchsize]
            idx = [q.index for q in query_samples]
            query_id = [q.id for q in query_samples]
            feed = self.ds.make_batch(self.model, idx)
            self.run_one_item((query_id, idx, feed))

            # count stats
            stats[len(idx)] += 1

    def enqueue(self, query_samples):
        with self.cv:
            scheduled = len(self.query_samples)
            # add new items to the queue
            self.query_samples.extend(query_samples)
            # notify only if queue was empty
            if scheduled == 0:
                self.cv.notify(n=1)

    def finish(self):
        # exit all threads
        self.done = True
        for worker in self.workers:
            with self.cv:
                self.cv.notify()
        for worker in self.workers:
            worker.join()


def parse_mlperf_log(log_path, args, csv, backend, acc_result, time_taken):
    rt = {}
    is_valid = False
    fname = os.path.join(log_path, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                rt[m.group(1).strip()] = m.group(2).strip()
    fmt1 = "filter,mode,time,name,scenario,qps,mean,min,max,50pt,90pt,95pt,99pt,99.9pt,valid,perf_ok,mindur_ok,minqs_ok,backend_name,backend_version,time_taken,model,precision,task,parameters(M),batchsize\n"
    fmt2 = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1f},{}\n"
    first = not os.path.exists(csv)

    def ns(n):
        return "{:.2f}".format(float(rt[n]) / 1000000.)

    def yes_no(n):
        if n.lower() in ["yes", "true", "valid"]:
            return 1
        return 0

    def metric(scenario):
        if scenario in ["Offline"]:
            return "Samples per second"
        if scenario in ["SingleStream"]:
            return "QPS w/o loadgen overhead"
        if scenario in ["Server"]:
            return "Scheduled samples per second"
        return None

    qps = rt.get(metric(args.scenario))
    with open(csv, "a") as f:
        if first:
            f.write(fmt1)
        if acc_result:
            f.write(fmt2.format(0, "acc", int(time.time()), args.name, args.scenario, acc_result,
                                "", "", "", "", "", "", "", "", "", "", "", "", backend.name(), backend.version(),
                                time_taken, args.model))
        else:
            f.write(fmt2.format(0, "perf", int(time.time()), args.name, args.scenario, qps,
                                ns('Mean latency (ns)'), ns('Min latency (ns)'), ns('Max latency (ns)'),
                                ns('50.00 percentile latency (ns)'), ns('90.00 percentile latency (ns)'),
                                ns('95.00 percentile latency (ns)'), ns('99.00 percentile latency (ns)'),
                                ns('99.90 percentile latency (ns)'),
                                is_valid, yes_no(rt.get('Performance constraints satisfied', "YES")),
                                yes_no(rt['Min duration satisfied']), yes_no(rt['Min queries satisfied']),
                                backend.name(), backend.version(), time_taken, args.model, args.precision, args.task, backend.parameters / 1e6, args.batchsize))

    if acc_result:
        print(f"result: {args.name}, {acc_result}")
    else:
        print(f"result: {args.scenario}, {backend.name()}, {args.name}, qps={qps}, mean={ns('Mean latency (ns)')}ms, 99={ns('99.00 percentile latency (ns)')}ms, took={time_taken:.1f}sec, valid={is_valid}")


def split_arg_to_kwargs(name, kwargs):
    cols = name.split(":")
    if len(cols) > 1:
        name = cols[0]
        for c in cols[1:]:
            c1 = c.split("=")
            if len(c1) > 1:
                kwargs[c1[0]] = c1[1]
    return name


def get_model_and_backend(args, tokenizer):
    backend = None
    kwargs = {}
    name = args.model
    task = args.task.lower()
    backend_name = split_arg_to_kwargs(args.backend, kwargs)
    model_class = "not_supported"

    kwargs['precision'] = args.precision

    if task == "pretraining":
        model = transformers.AutoModelForPreTraining.from_pretrained(name)
    elif task == "causualm":
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
        model_class = "AutoModelForCausalLM"
    elif task == "base":
        model = transformers.AutoModel.from_pretrained(name)
        model_class = "AutoModel"
    elif task == "lm":
        model = transformers.AutoModelWithLMHead.from_pretrained(name)
        model_class = "AutoModelWithLMHead"
    elif task == "masklm":
        model = transformers.AutoModelForMaskedLM.from_pretrained(name)
    elif task == "seq2seqlm":
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
    elif task in ["summary", "translate"]:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
        kwargs["generative"] = True
    elif task == "classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(name)
        model_class = "AutoModelForSequenceClassification"
    elif task == "multiplechoise":
        model = transformers.AutoModelForMultipleChoice.from_pretrained(name)
    elif task == "nextsentence":
        model = transformers.AutoModelForNextSentencePrediction.from_pretrained(name)
    elif task == "tokenclassification":
        model = transformers.AutoModelForTokenClassification.from_pretrained(name)
    elif task == "questionanswering":
        model = transformers.AutoModelForQuestionAnswering.from_pretrained(name)
        model_class = "AutoModelForQuestionAnswering"
    elif task == "tablequestionanswering":
        model = transformers.AutoModelForTableQuestionAnswering.from_pretrained(name)
    elif task == "imageclassification":
        model = transformers.AutoModelForImageClassification.from_pretrained(name)
    elif task == "vision2seq":
        model = transformers.AutoModelForVision2Seq.from_pretrained(name)
    elif task == "visualquestionanswering":
        model = transformers.AutoModelForVisualQuestionAnswering.from_pretrained(name)
    elif task == "audioclassification":
        model = transformers.AutoModelForAudioClassification.from_pretrained(name)
    elif task == "audioframeclassification":
        model = transformers.AutoModelForAudioFrameClassification.from_pretrained(name)
    elif task == "ctc":
        model = transformers.AutoModelForCTC.from_pretrained(name)
    elif task == "speechseq2seq":
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(name)
    elif task == "audioxvector":
        model = transformers.AutoModelForAudioXVector.from_pretrained(name)
    elif task == "maskimagemodeling":
        model = transformers.AutoModelForMaskedImageModeling.from_pretrained(name)
    elif task == "objectdetection":
        model = transformers.AutoModelForObjectDetection.from_pretrained(name)
    elif task == "imagesegmentation":
        model = transformers.AutoModelForImageSegmentation.from_pretrained(name)
    elif task == "semanticsegmentation":
        model = transformers.AutoModelForSemanticSegmentation.from_pretrained(name)
    elif task == "instancesegmentation":
        model = transformers.AutoModelForInstanceSegmentation.from_pretrained(name)
    else:
        raise NotImplementedError(f"model {name} not supported")

    max_input_size = tokenizer.max_model_input_sizes[name] if name in tokenizer.max_model_input_sizes else 1024

    if backend_name == "pytorch":
        backend = BackendPytorch(model, tokenizer, max_input_size, kwargs)
    elif backend_name == "deepspeed":
        backend = BackendDeepspeed(model, tokenizer, max_input_size, kwargs)
    elif backend_name == "onnxruntime":
        backend = BackendOnnxruntime(model, tokenizer, name, model_class, max_input_size, kwargs)
    else:
        raise NotImplementedError(f"backend {backend_name} not supported")
    return backend


def get_tokenizer(args):
    name = args.model
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return tokenizer


class DataSet:
    def __init__(self, backend, dataset_name, tokenizer, task, count):
        self.backend = backend
        self.dataset_name = dataset_name
        self.count = count
        self.tokenizer = tokenizer
        self.task = task
        self.samples = {}

    def load_query_samples(self, sample_ids):
        raise NotImplementedError("load_query_samples not implemented")

    def unload_query_samples(self, sample_ids):
        self.samples = {}

    def make_batch(self, backend, sample_ids):
        fmt = backend.tensor_format()
        if len(sample_ids) == 1:
            # fast path ... return a single sample
            feed = self.samples[sample_ids[0]]
            if backend.cuda and fmt != "np":
                for k, v in feed.items():
                    feed[k] = v.cuda()
            # print(feed['input_ids'].shape)
            return feed
        keys = list(self.samples[sample_ids[0]])
        values = []
        for k in keys:
            if backend.cuda and fmt != "np":
                v = [self.samples[idx][k].cuda() for idx in sample_ids]
            else:
                v = [self.samples[idx][k] for idx in sample_ids]
            values.append(v)
        batch = {}
        if fmt == "pt":
            for k, v in zip(keys, values):
                batch[k] = torch.cat(v)
        else:
            for k, v in zip(keys, values):
                batch[k] = np.concatenate(v)
        return batch

    def get(self, idx):
        return self.samples[idx]


class SquadDataSet(DataSet):
    def __init__(self, backend, dataset_name, tokenizer, task, count):
        super().__init__(backend, dataset_name, tokenizer, task, count)
        start = time.time()
        self.dataset = load_dataset(dataset_name, split="validation")
        log.info("loaded {} content items, took={:.1f}sec".format(len(self.dataset), time.time() - start))

    def load_query_samples(self, sample_ids):
        question = []
        text = []
        tensor_format = self.backend.tensor_format()
        for idx in sample_ids:
            sample = self.dataset[idx]
            if self.task == "summary":
                text.append("summary: " + sample["context"])
            elif self.task == "translation":
                text.append("Translate English to German: " + sample["context"])
            elif self.task in ["lm", "causualm"]:
                text.append(sample["context"])
            elif self.task == "questionanswering":
                question.append(sample["question"])
                text.append(sample["context"])

        if self.task == "questionanswering":
            tokens = self.tokenizer(question, text, return_tensors=tensor_format, padding=True)
        else:
            max_length = self.backend.max_input_size
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokens = self.tokenizer(text, return_tensors=tensor_format, padding=True, truncation=True, max_length=max_length)

        keys = list(tokens.keys())
        self.samples = {}
        log.info(f"squad dataset: {[i.shape for i in tokens.values()]}")
        if tensor_format == "pt":
            for i, idx in enumerate(sample_ids):
                self.samples[idx] = {key: torch.unsqueeze(tokens[key][i], dim=0) for key in keys}
        else:
            for i, idx in enumerate(sample_ids):
                self.samples[idx] = {key: np.expand_dims(tokens[key][i], axis=0) for key in keys}


class WikiTextDataSet(DataSet):
    def __init__(self, backend, dataset_name, tokenizer, task, count):
        super().__init__(backend, dataset_name, tokenizer, task, count)
        start = time.time()
        pat = re.compile(r"^\s=\s\w.*")
        ds = load_dataset(dataset_name, "wikitext-2-v1", split="validation")
        self.dataset = []
        buf = []
        for i, text in enumerate(ds):
            t = text['text']
            m = pat.search(t)
            if m:
                if buf:
                    self.dataset.append("".join(buf))
                    if len(self.dataset) > count:
                        break
                    buf = []
            if len(t):
                buf.append(t)

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        log.info("loaded {} content items, took={:.1f}sec".format(len(self.dataset), time.time() - start))

    def load_query_samples(self, sample_ids):
        text = []
        tensor_format = self.backend.tensor_format()
        max_idx = len(self.dataset)
        for idx in sample_ids:
            sample = self.dataset[idx % max_idx]
            if self.task == "summary":
                text.append("summary: " + sample)
            elif self.task == "translation":
                text.append("Translate English to German: " + sample)
            else:
                text.append(sample)
        max_length = self.backend.max_input_size
        max_length = 256
        tokens = self.tokenizer(text, return_tensors=tensor_format, padding=True, truncation=True, max_length=max_length)
        log.info(f"wikitext dataset: {[i.shape for i in tokens.values()]}")

        self.samples = {}
        keys = list(tokens.keys())
        if "decoder_input_ids" in self.backend.dummy_inputs.keys():
            input_mapping = self.backend.input_mapping or {}

            def dup_input_keys(i):
                d = {}
                if tensor_format == "pt":
                    for key in keys:
                        v = tokens[key][i]
                        d[input_mapping.get(key, key)] = torch.unsqueeze(v, dim=0)
                    d['decoder_input_ids'] = self.backend.dummy_inputs['decoder_input_ids'][:1]
                else:
                    for key in keys:
                        v = tokens[key][i]
                        d[input_mapping.get(key, key)] = np.expand_dims(v, axis=0)
                    d['decoder_input_ids'] = self.backend.dummy_inputs['decoder_input_ids'][:1].numpy()
                return d

            for i, idx in enumerate(sample_ids):
                self.samples[idx] = dup_input_keys(i)
        else:
            if tensor_format == "pt":
                for i, idx in enumerate(sample_ids):
                    self.samples[idx] = {key: torch.unsqueeze(tokens[key][i], dim=0) for key in keys}
            else:
                for i, idx in enumerate(sample_ids):
                    self.samples[idx] = {key: np.expand_dims(tokens[key][i], axis=0) for key in keys}


def get_dataset(args, backend, tokenizer, count):
    # https://huggingface.co/docs/datasets/use_with_pytorch
    if args.dataset in "squad":
        return SquadDataSet(backend, args.dataset, tokenizer, args.task, count)
    elif args.dataset in "wikitext2":
        return WikiTextDataSet(backend, args.dataset, tokenizer, args.task, count)
    raise NotImplementedError(args.dataset + " not implemented")


def main():
    args = get_args()

    # config.yaml
    here = os.path.dirname(os.path.abspath(__file__))

    # setup output directory
    mode = 'accuracy' if args.accuracy else 'performance'
    output_dir = os.path.abspath(os.path.join(args.output, args.model, args.backend, args.scenario, mode))
    os.makedirs(output_dir, exist_ok=True)

    # initial loadgen setup
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings
    # load mlperf rules
    settings = lg.TestSettings()
    mlperf_conf = args.mlperf_config
    if not mlperf_conf:
        mlperf_conf = os.path.join(here, "mlperf.conf")
    if not os.path.exists(mlperf_conf):
        print(mlperf_conf, "not found")
        return 1
    settings.FromConfig(mlperf_conf, args.model_task, args.scenario)

    count = args.count
    if not count and not args.accuracy:
        count = 1024

    #
    # from here the results directory is the current working dir !!!
    #
    os.chdir(output_dir)

    tokenizer = get_tokenizer(args)
    backend = get_model_and_backend(args, tokenizer)
    ds = get_dataset(args, backend, tokenizer, count)

    post_process = PostProcessNone()

    scenario = SCENARIO_MAP[args.scenario]
    if not args.batchsize:
        if scenario == lg.TestScenario.Server:
            print("Server: no batchsize, defaulting to 4")
            args.batchsize = 4
        else:
            args.batchsize = 8

    runner_map = {
        lg.TestScenario.SingleStream: RunnerBase,
        lg.TestScenario.Server: BatchingQueueRunner,
        lg.TestScenario.Offline: QueueRunner
    }
    runner = runner_map[scenario](backend, ds, args.threads, post_proc=post_process, max_batchsize=args.batchsize)

    def issue_queries(query_samples):
        runner.enqueue(query_samples)

    def flush_queries():
        pass

    # warmup
    ds.load_query_samples([0])
    for _ in range(5):
        feed = ds.make_batch(backend, [0])
        _ = backend.predict(feed)

    # get some estimate to set qps
    start = time.time()
    n = 20
    for _ in range(n):
        feed = ds.make_batch(backend, [0])
        _ = backend.predict(feed)
    guess = (time.time() - start) / n

    ds.unload_query_samples(None)

    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if scenario == lg.TestScenario.SingleStream and not args.max_latency:
        # guess the latency so there is enough query buffer
        args.max_latency = guess

    if scenario == lg.TestScenario.Server:
        settings.server_coalesce_queries = True
        # settings.server_num_issue_query_threads = args.threads
        if not args.qps:
            # Server requires --qps. If not given, guess it 1/3 of single reqest
            args.qps = str(int(1 / guess / 3))
            log.info("Server: no --qps, guessing %s", args.qps)

    if scenario == lg.TestScenario.Offline:
        if not args.qps:
            # Offline requires --qps. If not given, guess it 4 * of single reqest
            args.qps = str(int(4 / guess))
            log.info("Offline: no --qps, guessing %s", args.qps)

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        # settings.single_stream_target_qps = qps
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if args.queries:
        settings.min_query_count = int(args.queries)
        settings.max_query_count = int(args.queries)

    if args.max_latency:
        latency = int(args.max_latency * NANO_SEC)
        settings.single_stream_expected_latency_ns = latency
        settings.server_target_latency_ns = latency

    if args.max_latency_percentile:
        percentile = args.max_latency_percentile
        settings.single_stream_target_latency_percentile = percentile
        settings.server_target_latency_percentile = percentile

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(count, count, ds.load_query_samples, ds.unload_query_samples)

    log.info("starting {}".format(scenario))

    start_time = time.time()
    runner.start_run(args.accuracy)
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    time_taken = time.time() - start_time

    if args.accuracy:
        acc_result = ds.accuracy(output_dir)
    else:
        acc_result = None

    csv = args.csv or os.path.join(args.output, "summary.csv")

    parse_mlperf_log(output_dir, args, csv, backend, acc_result, time_taken)

    return 0


if __name__ == "__main__":
    sys.exit(main())
