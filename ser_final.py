import pickle
import socket
import struct
import numpy as np
import select  # Import select module for non-blocking I/O

class Server:
    def __init__(self, host='127.0.0.1', port=65432, num_clients=4):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.avg_logits = None
    
    def receive_data(self, CommunicationRounds=1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(self.num_clients)  # Listen for all clients
            
            print(f"Server is listening on {self.host}:{self.port}...")
            
            connections = []
            for _ in range(self.num_clients):
                conn, addr = s.accept()
                connections.append(conn)
                print(f"Client connected: {addr}")
            round_index = 0
            while round_index < CommunicationRounds:
                for batch_idx in range(9):  # Example: Process 10 batches
                    linear_outputs = []
                    
                    for conn in connections:
                        linear_output = self.receive_linear_outputs(conn)
                        linear_outputs.append(linear_output)

                    self.avg_logits = self.calculate_average_logits(linear_outputs)
                    # print(self.avg_logits.shape)
                    for conn in connections:
                        self.send_average_logits(conn, self.avg_logits)

                    print(f"Average logits sent to all clients for batch {batch_idx + 1}")

                    # Wait for signal from clients
                self.wait_for_signal(connections)
                print(f"Communication round {round_index + 1} completed.")
                round_index+=1
               
                    

            # Close connections
            for conn in connections:
                conn.close()
            print("All connections closed.")

    def receive_linear_outputs(self, conn):
        received_logits = []
        buffer = b''  # Initialize an empty buffer
        data = conn.recv(8)  # Receive shape information
        shape = struct.unpack('ii', data)
        num_elements = shape[0] * shape[1]

        while True:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data
            while len(buffer) >= 4:
                logit, buffer = struct.unpack('f', buffer[:4]), buffer[4:]
                if logit == (0.0,):  # Termination signal received
                    return np.array(received_logits).reshape(shape)
                received_logits.append(logit[0])

    def calculate_average_logits(self, logits_list):
        if not logits_list:
            return []

        stacked_logits = np.stack(logits_list, axis=0)
        avg_logits = np.mean(stacked_logits, axis=0)
        print(avg_logits.shape)
        return avg_logits

    def send_average_logits(self, conn, avg_logits):
        # Check for NaN values
        array_serialized = pickle.dumps(avg_logits)
        array_length = len(array_serialized)
        conn.sendall(struct.pack('!I', array_length))  # Send the length of the serialized array
        conn.sendall(array_serialized)

    def wait_for_signal(self, connections):
        # ready_count = 0
        # while ready_count < len(connections):
        #     readable, _, _ = select.select(connections, [], [], 0.1)  # Non-blocking check for readable sockets
        #     for conn in readable:
        #         try:
        #             signal = conn.recv(1024)
        #             if signal.strip() == b'ready':  # Check for the correct signal
        #                 print("Received ready signal from client.")
        #                 ready_count += 1
        #         except Exception as e:
        #             print(f"Error while waiting for signal: {e}")

        # if ready_count == len(connections):
        #     print("All clients are ready to proceed.")
        #     # return 1
        # else:
        #     print("Not all clients are ready.")
        #     # return 2
        for conn in connections:
            try:
                signal = conn.recv(1024)
                if signal == b'ready':
                    print("Received ready signal from client.")
                else:
                    print(f"Received unexpected signal: {signal}")
            except Exception as e:
                print(f"Error while waiting for signal: {e}")

if __name__ == "__main__":
    server = Server()
    server.receive_data(CommunicationRounds=40)  # Example: 2 communication rounds
