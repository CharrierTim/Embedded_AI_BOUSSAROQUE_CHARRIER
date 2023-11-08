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
    output = serial_port.read(12)
    print(f"Output: {hex(int.from_bytes(output, 'big'))}")

    # Create an empty list to store the extracted float values
    float_values = []

    # Iterate through the 12 bytes, splitting them into three 32-bit integers
    for i in range(0, 12, 4):
        # Extract 4 bytes
        four_bytes = output[i:i+4]

        # Reverse the order of the bytes
        reversed_bytes = four_bytes[::-1]

        # Convert the reversed bytes to a float
        float_value = struct.unpack('f', reversed_bytes)[0]

        # Add the float value to the list
        float_values.append(float_value)

    # Print the extracted float values
    for value in float_values:
        print(value)


def evaluate_model_on_STM32(iterations, serial_port):
    error = 0
    for i in range(iterations):
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        error += np.mean(abs(Y_test[i]-output)) / iterations
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"----------------------- Error: {np.mean(error):.2f}\n")
    return error


if __name__ == '__main__':
    X_test, Y_test = np.load(
        "./dataset/wine_quality_X_test.npy"), np.load("./dataset/wine_quality_Y_test.npy")

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        error = evaluate_model_on_STM32(10, ser)
