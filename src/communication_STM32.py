import serial

def synchronise_UART():
    with serial.Serial("/dev/tty.usbmodem1412403", 115200, timeout=1) as ser: # COM5 for H743 (nucleo) and COM6 for F411 (Nucleo)
        is_synced = False
        while (not is_synced):
            # Send 0xAB
            ser.write(b"\xAB")
            ret = ser.read(1)
            if (ret == b"\xCD"):
                is_synced = True
                print("Synchronised")

if __name__ == '__main__':
    print("Synchronising...")
    synchronise_UART()