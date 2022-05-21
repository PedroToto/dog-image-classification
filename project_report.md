# Report: DogImages Classification using AWS SageMaker
### Philippe Jean Mith

## Hyperparameter Tuning
### What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.
For this experimentation i used the densenet201 pretrained model because DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. And i used three hyperparameters.
| Hyperparameter | Range |
| -------------- | ----- |
| Learning rate  | (0.001, 0.1) |
| Batch Size     | [8, 32, 64] |
| Epochs         | (2, 5) |

## Debugging and Profiling
For model Debugging i configured the hook in SageMaker's python SDK using the Estimator class and instantiate it with with `create_from_json_file()` from `smdebug.pytorch.Hook`. I added the *SMDebug hook* for PyTorch with TRAIN mode in the train function, EVAL mode in the test function and specified the debugger rules and configs in what i set a save interval of 100 and 10 for training and testing respectively. For the Profiling, i specified the metrics to track and create the profiler rules. 

## Results
They are some anomalous behaviour in the debugging output. `LossNotDecreasing`, `Overtraining` metrics gave return Error and ProfilerReport i will remove the `loss_not_decreasing`, `ProfilerReport` metrics in the debugger rules.

## Model Deployment
To deploy the mode i call the `deploy` method from the estimator with the instance type with the parameter `instance_type` and the number of instances with the oarameter `initial_instance_count`.

![Endpoints](https://github.com/PedroToto/dogImages_classification/blob/main/Endpoints.PNG)

## Standout Suggestions
After the model deployement i had some issue to predict on the endpoint.
The first error was `ModuleNotFoundError: No module named 'smdebug'`. To work around this is issue, i created a different script for deployment which was recommended as a solution for the issue. After that there was no more `ModuleNotFoundError: No module named 'smdebug'` error. However i couldn't still have the result from the prediction. When i check in the Log groups what i found is in the picture bellow.
![error](https://github.com/PedroToto/dogImages_classification/blob/main/Error3.PNG)
The status code 200 means that the request is fulfilled but i wanted the prediction result differentely.
