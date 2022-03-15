from core import Config, Experiment

conf = Config("./configs/example.yaml")
exp = Experiment(conf)

exp.run()