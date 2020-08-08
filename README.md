# Simple neptune.ai Logger
This repository provides simple but useful python class to manage experiments using [neptune.ai](https://neptune.ai).
[neptune.ai](https://neptune.ai) is a simple but powerful experiment management service. In this repository, I provide a simple logger class to log metrics, images, files, artifacts and hyper-parameters for any experiment. </br>
A blog post I wrote about this can be found [here](https://yoonki-j.info/neptune-ai-intro/). (Written in Korean)

# How-to-Use
[NeptuneLogger](Neptune.py/NeptuneLogger.py) is implemented with python and neptune-client. Therefore, it can be intagrated with any project or experiment you would like to do. Detailed and step-by-step starter can be found [here](https://docs.neptune.ai/learn-about-neptune/ui.html).

## Install ```neptune-client```
You can install neptune using pip. (it's not **neptune**, but **neptune-client**)</br>
```pip install neptune-client```

If you want for neptune to inspect your hardware resource, also install ```psutil``` </br>
```pip install neptune-client psutil```

If you want to run the sample MNIST codes with neptune, you have to install:
* torch
* torchvision
* matplotlib

## Prepare your ```NEPTUNE_API_TOKEN```
Once you sign and log in to neptune, you will get the unique token for API. </br>
If you don't know or remember, try to follow 'Getting Started' in upper right corner of the neptune page.

You can specify API token directly in the code, but it's unsafe. </br>
I recommend setting API token as an environment variable. </br>
For example, on linux you can add it in ```~/.bashrc``` or use command as below: </br>
```export NEPTUNE_API_TOKEN = YOUR_API_TOKEN```

## Log whatever you like!
Define logger using ```NeptuneLogger``` and log!
To initiate ```NeptuneLogger```, there are many arguments to be specified.

* ```api_key```: Your API token
* ```project_name```: Project name with the format of ```YOUR_ID/PROJECT_NAME```. This can be found in your main neptune page when you log in.
* ```experiment_name```: Arbitrary name of the experiment (e.g. SampleMNIST)
* ```description```: Short description of the experiment
* ```tags```: List of tags you want to specify. Tags are useful to filter your experiments of the project!
* ```hparams```: Hyper-parameters as python dictionary.
* ```upload_source_files```: Any source code files to upload.
* ```hostname```: Hostname. Useful when you experiment with different devices or servers.
* ```offline```: True/False. Log online neptune page if True. Do not log online otherwise.

## Run sample MNIST experiment
For those who want to try out how neptune works, I provide simple MNIST classification experiment. In ```main.py```, modify arguments for ```NeptuneLogger``` with your own. Then, run it and go neptune page to see what happens.

You will see metrics, plots, hyper-parameters are logged succesfully! </br>