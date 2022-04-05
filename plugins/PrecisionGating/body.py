import logging

from core.plugin import Plugin
from core.registry import registerPlugin
from .pg_utils import *


@registerPlugin
class precisionGating(Plugin):
    @property
    def pluginName(self):
        return "PrecisionGating"

    def __init__(self, wbits, abits, pgabits, threshold, n_banks = 8, sched_th = 4,
                ena_schedule = False, sparse_bp=False, skip_layers=None):
        self.wbits = wbits
        self.abits = abits
        self.pgabits = pgabits
        self.sparse_bp = sparse_bp
        self.threshold = threshold
        self.skip_layers = skip_layers
        self.n_banks = n_banks
        self.sched_th = sched_th
        self.ena_schedule = ena_schedule

        self.cmd_logger = logging.getLogger("PG")

        self.cnt_out = {}
        self.cnt_high = {}
        self.sparsity = 0.0

    def loadState(self, checkpoint_path=''):
        """
        No State to save
        """
        pass

    def saveState(self, checkpoint_path=''):
        """
        No State to load
        """
        pass

    # We use model creation hook here for loading the checkpoint after creating the model
    def modelCreationHook(self, model):
        replacePGModule(model, wbits=self.wbits, abits=self.abits, pgabits=self.pgabits,
                        th=self.threshold, sparse_bp=self.sparse_bp)
        #Eliminate Distillation Head of levit
        if hasattr(model, "head_dist"):
            del model.head_dist
            model.head_dist = None

        # Initilize counter for the sparsity
        for m, n in model.named_modules():
            if hasattr(m, 'weight_fp'):
                self.cnt_out[n] = 0
                self.cnt_high[n] = 0

    # Copy full precision weight for update
    def preUpdateHook(self, model, inputs, targets, loss, iter_id):
        for p in model.modules():
            if hasattr(p, 'weight_fp'):
                p.weight.data.copy_(p.weight_fp)

    def iterTailHook(self, model, inputs, targets, logger, iter_id):
        for p in model.modules():
            if hasattr(p, 'weight_fp'):
                p.weight_fp.data.copy_(p.weight.data.clamp_(-1, 1))

    def epochHeadHook(self, model, epoch_id):
        # Clear the counter at the begining of each epoch
        self.sparsity = 0.0
        for key in self.cnt_out.keys():
            self.cnt_out[key] = 0
            self.cnt_high[key] = 0

    def evalIterHook(self, model, iter_id, logger=None):
        for n, m in model.named_modules():
            if isinstance(m, (PGConv2d, PGAttention)):
                self.cmd_logger.debug("Layer {0}, out: {1}, high: {2}".format(n, m.num_out, m.num_high))
                if n not in self.cnt_out.keys():
                    self.cnt_out[n] = m.num_out
                    self.cnt_high[n] = m.num_high
                else:
                    self.cnt_out[n] += m.num_out
                    self.cnt_high[n] += m.num_high

    def evalTailHook(self, model, epoch_id=None, logger=None):
        if sum(self.cnt_high.values()) > 0:
            self.sparsity = 100 - (sum(self.cnt_high.values()) * \
                            1.0 / sum(self.cnt_out.values())) * 100
            self.cmd_logger.info('Sparsity of the update phase: {0:.2f}'.format(self.sparsity))

            # If it is during training
            if epoch_id is not None and logger is not None:
                logger.log_scalar(self.sparsity, "PG Sparsity", "Test", epoch_id)
