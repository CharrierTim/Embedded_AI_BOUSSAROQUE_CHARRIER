import serial
import struct
import numpy as np

PORT = "COM4"


def synchronise_UART(serial_port):
    while (1):
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if (ret == b"\xCD"):
            break


def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float32)  # Convert inputs to float16
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)


def read_output_from_STM32(serial_port):
    output = serial_port.read(3)
    print(f"Raw output: {output}")

    float_values = [int(out)/255 for out in output]
    {print(val) for val in float_values}
    return float_values


def evaluate_model_on_STM32(iterations, serial_port):
    accuracy = 0
    for i in range(iterations):
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        if (np.argmax(output) == np.argmax(Y_test[i])):
            accuracy += 1 / iterations
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"----------------------- Accuracy: {accuracy:.2f}\n")
    return error


if __name__ == '__main__':
    X_test, Y_test = np.load("./dataset/wine_quality_X_test.npy"), np.load("./dataset/wine_quality_Y_test.npy")

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        error = evaluate_model_on_STM32(10, ser)
