
# Wine Quality Prediction

**Table of Contents:**

- [Wine Quality Prediction](#wine-quality-prediction)
  - [Authors](#authors)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Wine Quality Prediction](#wine-quality-prediction-1)
    - [The dataset](#the-dataset)
    - [The model](#the-model)
    - [Communication with the STM32](#communication-with-the-stm32)
  - [The C code for the AI implementation](#the-c-code-for-the-ai-implementation)
    - [Syncronization](#syncronization)
    - [Acquiring and pre-processing the data](#acquiring-and-pre-processing-the-data)
    - [Post-processing the data](#post-processing-the-data)
  - [Results of the classifier](#results-of-the-classifier)
  - [Adversarial attack on the classifier](#adversarial-attack-on-the-classifier)

## Authors

- [Timothée Charrier](https://github.com/CharrierTim)
- [Alexis Boussaroque](https://github.com/H3Aether)

## Overview

An embedded AI project that aims to implement Wine Quality Prediction on the STM32L4R9AI platform.

## Project Structure

```bash
project-root/
│
├── dataset/
│ ├── wine_quality_X_test.npy
│ ├── wine_quality_Y_test.npy
│ ├── wine_quality_X_test_attacked_0.05.npy
│ ├── ...
│ ├── wine_quality_X_test_attacked_1.npy
│ └── winequalityN.csv
│
├── model/
│ └── wine_quality_classifier.h5
│ 
├── src/
│ ├── adversarial_example_attack.ipynb
│ ├── algorithms_comparison.py
│ ├── app_x-cube-ai.c
│ ├── communication_STM32.py
│ ├── dataset_handling.py
│ ├── wine_quality_classifier.ipynb
│ └── 
└── README.md
```

## Prerequisites

- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html)
- [Python 3.8](https://www.python.org/downloads/release/python-380/)

## Getting Started

Follow these steps to set up and run the project:

### Installation

1. Connect your STM32 development board to your computer.

2. Install the required Python packages: serial, numpy, csv, matplotlib and tensorflow

    ```bash
    pip install serial numpy csv matplotlib tensorflow
    ```

3. Open STM32CubeIDE and import the project.

4. Upload the project to your development board.

5. Run the Python script to start the prediction.

    ```bash
    python3 communication_STM32.py
    ```

Let's delve into the details of the model and the communication.

## Wine Quality Prediction

### The dataset

The objective of the dataset is to evaluate the quality of a wine based on 12 criteria such as acidity, sugar, pH... The result is a score between 3 and 9 (out of 10). But the dataset is not balanced, so we decided to reduce the number of categories. Here is the initial distribution of the scores:

![Wine Quality Raw Dataset Repartion](./img/initial_label_distribution.png)

To deal with this problem, we decided to augment to the dataset by adding new samples with random noise to the existing ones. You can find our own implementation of this method in `src/dataset_handling.py`.

For better results, we also reduced to 3 categories: bad, average, and good. Here is the new distribution after the remapping and data augmentation:

![Wine Quality Dataset Repartion](./img/label_distribution_after_data_aug.png)


To handle the dataset, we created a Python Class called `Dataset` that can be found in `src/dataset_handling.py`. This class is used to load the dataset, format, split and augment it. Here is a UML diagram of the class:

![PlantUML Diagram](https://kroki.io/plantuml/svg/eNqNksFOwzAMhu97Ch-LgD1AJaQhIQQXLnCvvNXdgpK4StxKBfHu2CmVNk3TOMVJfn-_7WSTBZMMwa92HnOGJxTMJPC9ArhvdQM1xCH00zq2mBJOen7bNC46aZqqc54iBqohS7oD0zc47ANFQXEcm8gu08MbR7pRkK2W33EKKNUF-bEy995JJZSl6RP3nExWQ-cZxXQy9L4Iz1hVYR1JC_JnFVkIktsfBLhb2q1ry1fOx4FKFzYCzIBz71A6X68otmD5lyDLXBT0qoFD774ogyj0T_OCsfWUgLeftBPYTpAIWxf3RbQ4d4mDett0QRNgnpeYzMn1Mma5FvFcgnzCLpaBR4NR6GWCEf2gVZrRjuNISZZ6PG7JazrDqNVyyte9y4Op9butp84uKkgS6oyUb3b2rhbr7T_QZ0-sNo_z9qxFbMtQyx-w-pfrI5eNRvrvfwHCbBDd)

### The model

The model is a Multi-Layer Perceptron, with 12 inputs (one for each parameter). The hidden layers are the following:

```text
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 20)                260       
                                                                 
 dropout (Dropout)           (None, 20)                0         
                                                                 
 dense_1 (Dense)             (None, 15)                315       
                                                                 
 dense_2 (Dense)             (None, 3)                 48        
                                                                 
=================================================================
Total params: 623 (2.43 KB)
Trainable params: 623 (2.43 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
The output is a vector of size 3 (bad, average good) in one hot encoding. 
The training can be found in `src/wine_quality_classifier.ipynb`.

### Communication with the STM32

The communication with the STM32 is done via UART and made of several steps:

- The laptop repeatdly sends 0xAB, waiting for the STM32 to answer 0xCD. Once this is done, the synchronization is complete and data can be sent.

- The laptop sends 12 floats of 32 bits (48 bytes in total) which are all between -1 and 1 (as the data is normalized).

- The STM32 processes the inputs and returns 3 floats. Each one corresponds the probability of the wine being bad, average, or good. The laptop picks the highest one and interprets it as the result.

## The C code for the AI implementation

Implementing this classifier was done using Cube-AI pakage. We have to provide the model in .h5 format and the input and validation data in .npy format. Then, we wrote a C code to synchronize the STM32, pre and post processing the data and finally, the prediction.

The source code can be found in `src/app_x-cube-ai.c`.

### Syncronization

The synchronization is done using a simple handshake. The STM32 sends a byte with the value 0xAB and waits for the PC to send back 0xCD. Once the PC receives the 0xAB byte, it sends back 0xCD and the synchronization is done.

```c
#define SYNCHRONISATION 0xAB
#define ACKNOWLEDGE 0xCD

void synchronize_UART(void)
  {
    bool is_synced = 0;
    unsigned char rx[2] = {0};
    unsigned char tx[2] = {ACKNOWLEDGE, 0};

    while (!is_synced)
    {
      HAL_UART_Receive(&huart2, (uint8_t *)rx, sizeof(rx), TIMEOUT);

      if (rx[0] == SYNCHRONISATION)
      {
        HAL_UART_Transmit(&huart2, (uint8_t *)tx, sizeof(tx), TIMEOUT);
        is_synced = 1;
      }
    }

    return;
  }
```

### Acquiring and pre-processing the data

Acquiring the data is done using the HAL_UART_Receive function. The data is received as an array of bytes. We have to reconstruct the floats from the bytes (4 bytes per float). This reconstruction is done using the following code:

```c	
#define BYTES_IN_12_FLOATS 48

int acquire_and_process_data(ai_i8 *data[])
  {
    //
    // 1. Variables for data acquisition
    //

    unsigned char tmp[BYTES_IN_12_FLOATS] = {0};
    int num_elements = sizeof(tmp) / sizeof(tmp[0]);
    int num_floats = num_elements / 4;

    //
    // 2. Receive data from UART
    //

    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t *)tmp, sizeof(tmp), TIMEOUT);

    // Check the return status of HAL_UART_Receive
    if (status != HAL_OK)
    {
      printf("Failed to receive data from UART. Error code: %d\n", status);
      return (1);
    }

    //
    // 3. Reconstruct floats from bytes
    //

    if (num_elements % 4 != 0)
    {
      printf("The array length is not a multiple of 4 bytes. Cannot reconstruct floats.\n");
      return (1);
    }

    for (size_t i = 0; i < num_floats; i++)
    {
      unsigned char bytes[4] = {0};

      // Reconstruction of the bytes
      for (size_t j = 0; j < 4; j++)
      {
        bytes[j] = tmp[i * 4 + j];
      }

      // Store the bytes in 'data'
      for (size_t k = 0; k < 4; k++)
      {
        ((uint8_t *)data)[(i * 4 + k)] = bytes[k];
      }
    }

    return (0);
  }
```

Once the data is acquired, Cube-AI will run the model and return the prediction. The prediction is an array of 12 bytes representing 3 floats (i.e. the 3 categories). This is the role of the post-processing function.

### Post-processing the data

We decided to convert to floats and multiply by 255 to get the probability in uint8_t format (to only send one byte per category). By scaling the probabilities to the range [0, 255], we keep a precision of 0.4%, which is more than enough for our application.


This is done using the following code:

```c
int post_process(ai_i8 *data[])
  {
    //
    // Get the output data
    //

    if (data == NULL)
    {
      printf("The output data is NULL.\n");
      return (1);
    }

    uint8_t *output = data;

    // An array to store the float outputs
    float outs[3] = {0.0};
    char outs_uint8[3] = {0};

    /* Convert the probability to float */
    for (size_t i = 0; i < 3; i++)
    {
      uint8_t temp[4] = {0};

      // Extract 4 bytes to reconstruct a float
      for (size_t j = 0; j < 4; j++)
      {
        temp[j] = output[i * 4 + j];
      }

      // Reconstruct the float from the bytes
      outs[i] = *(float *)&temp;

      // Convert the float to uint8_t for UART transmission
      outs_uint8[i] = (char)(outs[i] * 255);
    }

    //
    // Transmit the output data
    //

    HAL_StatusTypeDef status = HAL_UART_Transmit(&huart2, (uint8_t *)outs_uint8, sizeof(outs_uint8), TIMEOUT);

    // Check the return status of HAL_UART_Transmit
    if (status != HAL_OK)
    {
      printf("Failed to transmit data to UART. Error code: %d\n", status);
      return (1);
    }

    return 0;
  }
```
We also implemented an error handling function that will be called if the prediction fails due to a memory allocation error or a data acquisition error.

Then, the PC can receive the prediction, convert it and compare it to the ground truth.

## Results of the classifier

The classifier was performing as expected. The python model was able to predict the quality of the wine with an accuracy around 68%. The C code was able to predict the quality of the wine with pretty much the same accuracy. Here is the output accuracy and prediction of the classifier after 100 iterations:

```bash	
----- Iteration 100 -----
   Expected output: [0 0 1]
   Received output: [0.07058823529411765, 0.24705882352941178, 0.6745098039215687]
----------------------- Accuracy: 0.67
```

To reproduce these results, you can run the following command:

```bash
python3 communication_STM32.py
```

## Adversarial attack on the classifier

The adversarial attack is done with a 'white box' and follows the methdology described on the tensorflow website: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

We start by computing the gradient of the loss at a the input we want to attack:

```python
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input, label):
  input_tensor = tf.convert_to_tensor(input.reshape(1, 12), dtype=tf.float32)
  label_tensor = tf.convert_to_tensor(label.reshape(1, 3), dtype=tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    prediction = model(input_tensor)
    loss = loss_object(label_tensor, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_tensor)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
```

We define a budget (epsilon), then add the gradient to the input to create a biaised input:

```python
def create_adversarial_example(input, label, eps):
  perturbations = create_adversarial_pattern(input, label)
  adversarial_example = input + eps * perturbations
  return adversarial_example
```

Using the above function, we create a series of attacked datasets, with different budgets. We evaluate the model on these datasets. As expected the accuracy decreases. It can be noticed that it goes below 33%, which means the model became worse than random guessing.

Here is the accuracy of the model on the attacked datasets:

![Wine Quality Dataset Repartion](./img/effect_of_attack_budget_on_accuracy.png)

To reproduce these results, you have to modify the `src/comminication_STM32.py` file and set the `attack_model` variable to `True` and set the `bugdet` variable to one of the following values: 0.05, 0.10, 0.20, 0.30, 0.50, 0.80 and 1.00.

Then, you can run the following command:

```bash
python3 communication_STM32.py
```