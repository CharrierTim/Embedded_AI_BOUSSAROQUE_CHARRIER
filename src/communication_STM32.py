import serial
import numpy as np

PORT = "COM4"

def synchronise_UART(serial_port):
        while (1):
            serial_port.write(b"\xAB")
            ret = serial_port.read(1)
            if (ret == b"\xCD"):
                break

def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float16) # Convert inputs to float16
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)

def read_output_from_STM32(serial_port):
    output = serial_port.read(1)
    return output
    
def evaluate_model_on_STM32(iterations, serial_port):
    error = 0
    for i in range(iterations):
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        error += abs(Y_test[i]-output) / iterations
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"----------------------- Error: {error:.2f}\n")
    return error


if __name__ == '__main__':
    X_test, Y_test = np.load("./dataset/test_X.npy"), np.load("./dataset/test_Y.npy")

    with serial.Serial(PORT, 115200, timeout=1) as ser: # COM5 for H743 (nucleo) and COM6 for F411 (Nucleo)
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Sending inputs...")
        send_inputs_to_STM32(X_test[0], ser)
        print("Inputs sent")

        print("Reading output...")
        print(f"Received: {read_output_from_STM32(ser)}")
