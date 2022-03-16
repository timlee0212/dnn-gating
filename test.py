from core import Config, Experiment

conf = Config("configs/default.yaml")
exp = Experiment(conf)

exp.run()