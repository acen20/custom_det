from detectron2.engine.hooks import EvalHook
### Modified EvalHook ###

class EvalHook(EvalHook):
	def __init__(self,eval_period, eval_function):
		super(EvalHook,self).__init__(eval_period, eval_function)
		
	def _do_eval(self):
		result = self._func()
		return result
        
	def after_step(self):
		signal = None
		next_iter = self.trainer.iter + 1
		if self._period > 0 and next_iter % self._period == 0:
			# do the last eval in after_train
			if next_iter != self.trainer.max_iter:
				signal = self._do_eval()
		return signal

    
