from cgitb import text
import os
import shutil
from subprocess import PIPE, Popen

def safe_exec(command):
    result = "500 Server Error"
    while "500 Server Error" in result or "CUDA_LAUNCH_BLOCKING=1" in result:
        p = Popen(command, shell=True, stderr=PIPE, text=True)
        stdout, stderr = p.communicate()
        result = stderr

_model_list = [["bert-base-uncased", "bert_base_sanger_2e-3.json"],
               ["prajjwal1/bert-mini", "bert_mini_sanger_2e-3.json"],
               ["prajjwal1/bert-tiny", "bert_tiny_sanger_2e-3.json"]]
if not os.path.exists("./export"):
    os.mkdir("./export")

# print("Exporting checkpoints for CLOTH task...")
# for model in _model_list:
#     safe_exec("./scripts/eval_sparse_on_cloth.sh {0} {1}".format(model[0], model[1]))
#     shutil.move("dump.npy", "./export/{0}_CLOTH_0.002.npy".format(model[0].split("/")[-1]))

# glue_tasks = ["mrpc", "qqp", "sst2", "stsb", "cola", "mnli", "qnli", "rte"]
# print("Exporting checkpoints for GLUE task...")
# for model in _model_list:
#     for task in glue_tasks:
#         if not "base" in model and "cola" in task:
#             continue
#         print("Exporting {0} for task: {1}".format(model, task))
#         safe_exec("./scripts/eval_sparse_on_glue.sh {0} {1} {2}".format(task, model[0], model[1]))
#         shutil.move("dump.npy", "./export/{0}_GLUE_{1}_0.002.npy".format(model[0].split("/")[-1], task))

print("Exporting checkpoints for SQUAD task...")
for model in _model_list:
    safe_exec("./scripts/eval_sparse_on_squad.sh {0} {1}".format(model[0], model[1]))
    shutil.move("dump.npy", "./export/{0}_SQUAD_0.002.npy".format(model[0].split("/")[-1]))


