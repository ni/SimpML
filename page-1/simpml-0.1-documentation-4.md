# SimpML 0.1 documentation

```
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
cwd = Path.cwd()
ROOT_PATH = str(cwd.parent.parent.parent.parent)
sys.path.append(ROOT_PATH)
```

### Install SimpML[](broken-reference)

As the compiled SimpML package is hosted on Local Python Package Index (PyPI) you can easily install it with pip but first of need to make sure there is a mount:

#### mount[](broken-reference)

From: \opfiles:nbsphinx-math:_Public_:nbsphinx-math:[\`](broken-reference)DS \` To: YOUR\_PATH

Then, run the following command:

#### pip[](broken-reference)

pip install simpml –find-links YOUR\_PATH/niai

### SimpML’s applications all use the same basic steps and code:[](broken-reference)

* Create a DataManager with appropriate Pre-Processing Pipeline
* Create a ExperimentManager
* Call to run\_experiment method
* Get the model with the best performance
* Make predictions or view the results
* Create a Interpreter
* Get insights and see the interpretation of your model

In this quick start, we’ll show these steps for a wide range of difference applications and datasets. As you’ll see, the code in each case is extremely similar, despite the very different models and data being used.

```
from simpml.tabular.all import *
from simpml.vision.all import *
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC)
