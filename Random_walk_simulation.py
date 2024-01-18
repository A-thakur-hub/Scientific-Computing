import random
import matplotlib.pyplot as plt
from mpi4py import MPI


def move_object(obj, current_position, step, np, return_dict_list):
    r = random.randint(1, 2)  # Flip a coin 1-Tail 2-Head

    # Determine the new position
    if r == 1:
        new_position = current_position - 1 if current_position > 1 else N
        movement_direction = 'left'
    else:
        new_position = current_position + 1 if current_position < N else 1
        movement_direction = 'right'

    # Check if the new position is in a different partition
    target_partition = (new_position - 1) // (N // np) + 1
    current_partition = (current_position - 1) // (N // np) + 1

    # Send the object to the target partition if needed
    if target_partition != current_partition:
        lab_send(obj, new_position, target_partition)

    # Synchronize across partitions
    lab_barrier()

    # Receive any objects from other partitions
    received_objects = lab_probe()
    for received_obj, received_pos in received_objects:
        return_dict_list[rank][received_obj] = received_pos

    # Update the position in the current partition
    return_dict_list[rank][obj] = new_position

    # Print movement information
    print(f"Step {step}: {obj} moved {movement_direction}")



def simulate_random_walk(obj_positions, np, M):
    return_dict_list = [{} for _ in range(np)]
    return_dict_list[rank] = obj_positions.copy()

    for m in range(1, M + 1):
        for obj, current_position in obj_positions.items():
            move_object(obj, current_position, m, np, return_dict_list)

        # Synchronize across partitions after each step
        lab_barrier()

        # Update object positions
        obj_positions = return_dict_list[rank].copy()
    # Print final positions
    if rank == 0:
        print(f"Final Positions: {obj_positions}")

    return obj_positions




def plot_positions(positions, title):
    plt.figure(figsize=(8, 6))
    for obj, pos in positions.items():
        plt.scatter(pos, 0, label=obj, s=100)  # Plotting on a line, hence y-coordinate is 0
    plt.title(title)
    plt.xlabel('Grid Point')
    plt.xticks(range(1, N + 1))
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

if _name_ == '_main_':
    N = 10  # Number of grid points
    np = MPI.COMM_WORLD.Get_size()  # Number of partitions (workers)
    print("np", np)
    M = 4  # Number of steps

    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize positions randomly
    obj_positions = {f'Obj{i}': random.randint(1, N) for i in range(1, 6)}
    print(f"Initial Positions (Rank {rank}):", obj_positions)


    def lab_send(obj, pos, target_partition):
        if rank == target_partition - 1:
            comm.send((obj, pos), dest=target_partition)


    def lab_probe():
        received_objects = []
        for source in range(np):
            if source != rank:
                try:
                    received_obj, received_pos = comm.recv(source=source)
                    received_objects.append((received_obj, received_pos))
                except MPI.EOFError:
                    pass
        return received_objects


    def lab_barrier():
        comm.Barrier()


    # Simulate the random walk using message passing
    final_positions = simulate_random_walk(obj_positions, np, M)

    # Plotting
    plot_positions(obj_positions, f'Initial Positions (Rank {rank})')
    plot_positions(final_positions, f'Final Positions (Rank {rank})')

    if rank == 0:
        input("Press enter to exit")