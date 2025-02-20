from mpi4py import MPI
from logging_config import logger

# def set_comm_name(comm, name):
#     try:
#         keyval = MPI.Comm.Create_keyval()
#         comm.Set_attr(keyval, name)
#         return keyval
#     except Exception as e:
#         logger.error(f"Error in set_comm_name; {e}", exc_info=True)
#         return None

# def get_comm_name(comm, keyval):
#     try:
#         return comm.Get_attr(keyval)
#     except Exception as e:
#         logger.error(f"Error in get_comm_name; {e}", exc_info=True)
#         return None
    
def set_group_size(comm, num_groups):
    try:
        size = comm.Get_size()
        group_size = size // num_groups
        return group_size
    except Exception as e:
        logger.error(f"Error in set_group_size; {e}", exc_info=True)
        return None
    
def set_group(comm, group_size):
    try:
        rank = comm.Get_rank()
        group = rank // group_size
        return group
    except Exception as e:
        logger.error(f"Error in set_group_size; {e}", exc_info=True)
        return None