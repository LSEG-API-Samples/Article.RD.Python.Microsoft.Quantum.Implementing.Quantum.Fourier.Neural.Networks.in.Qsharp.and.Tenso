# Article.RD.Python.Microsoft.Quantum.Implementing.Quantum.Fourier.Neural.Networks.in.Qsharp.and.Tenso
## <a id="overview"></a>Overview
This article builds on our article "TensorFlow Variational Quantum Neural Networks in Finance". 
Within the DNN structure, we will be injecting a more complex layer structure that is implementing a Quantum Fourier Transform (QFT) on its inputs. 
We will  explore how we can implement a dynamic QFT layer, able to scale up to a variable number of Qubits, and how such a layer can be simulated locally or executed on Azure Quantum.

Details and concepts are further explained in the [Microsoft Quantum – Implementing Quantum Fourier Neural Networks in Q# and Tensorflow](https://developers.refinitiv.com/en/article-catalog/article/microsoft-quantum---implementing-quantum-discrete-fourier-transf.html) published on the [Refinitiv Developer Community portal](https://developers.refinitiv.com).

## <a id="disclaimer"></a>Disclaimer
The source code presented in this project has been written by Refinitiv only for the purpose of illustrating the concepts of creating example scenarios using the Refinitiv Data Library for Python.

***Note:** To [ask questions](https://community.developers.refinitiv.com/index.html) and benefit from the learning material, I recommend you to register on the [Refinitiv Developer Community](https://developers.refinitiv.com)*

## <a name="prerequisites"></a>Prerequisites

To execute any workbook, refer to the following:

- A Refinitiv Desktop license (Refinitiv Eikon or Refinitiv Workspace) that has API access 
- Tested with Python 3.7.13
- Packages: [Q# & DQK] (https://learn.microsoft.com/en-us/azure/quantum/overview-what-is-qsharp-and-qdk, 
			[numpy] (https://pypi.org/project/numpy/),
			[pandas](https://pypi.org/project/pandas/), 
			[tensorflow2](https://www.tensorflow.org/install), [refinitiv.data](https://pypi.org/project/refinitiv-data/),
			[sci-kit learn] (https://pypi.org/project/scikit-learn/),
			[Microsoft SDK] (https://dotnet.microsoft.com/en-us/download/visual-studio-sdks),
			[Azure CLI] (https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
- RD Library for Python installation:  '**pip install refinitiv-data**'


  
## <a id="authors"></a>Authors
* **Marios Skevofylakas**
