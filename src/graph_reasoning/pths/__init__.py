import os


def get_pth(pth_name):
    # Determine the path to the PTH file
    pth_path = os.path.join(os.path.dirname(__file__), f'{pth_name}.pth')

    return pth_path