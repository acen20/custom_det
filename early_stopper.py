from detectron2.evaluation import COCOEvaluator
import json, os
import torch
import numpy as np
def early_stopping(cfg, trainer, patience):
    # Calculate accuracy/AP
    cfg.DATASETS.TEST = ("custom_test",)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
    results = trainer.test(cfg, trainer.model, evaluators = [evaluator])
    new_AP = results['bbox']['AP50']

    # If new AP50 is "nan", it means the model has not learned anything, so we just return to training loop
    if np.isnan(new_AP):
      return
    
    model_file_name = "best_model.pth"
    
    # If best model file does not exist, save current as best model
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, model_file_name)) == False:
        torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, model_file_name))

    # current stats
    obj = {
      'model_name': model_file_name,
      'AP': new_AP
    }

    # check if there is a history of accuracies by checking if the file exists
    file_name = 'last_best_acc.json'
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, file_name)):
      
        # read previous accuracy
        with open (os.path.join(cfg.OUTPUT_DIR, file_name), 'r') as f:
            previous_stats = json.load(f)

        # get previous stats
        previous_AP = previous_stats['AP']
        previous_model_file_name = previous_stats['model_name']

        # if new accuracy is less than previous accuracy, wait and stop!!
        if new_AP < previous_AP:
            if trainer.patience_iter == patience:
                # rename best_model.pth to model_final.pth and stop training
                os.rename(os.path.join(cfg.OUTPUT_DIR, previous_model_file_name),
                         os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
                return True
            trainer.patience_iter += 1

        else: # continue training
            # reset patience_iter
            trainer.patience_iter = 0
            
            # save as best_model.pth
            print("Saving current model as best model")
            torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, model_file_name))
            
            # write current stats
            with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
                json.dump(obj, f)
                  
    else:
      with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
          json.dump(obj, f)