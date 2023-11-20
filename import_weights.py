import h5py

def explore_group(group, indent=0):
    # Iterate through items in the group
    for item_name, item in group.items():
        # Print indentation based on the level of nesting
        print(" " * indent + f"Name: {item_name}, Type: {type(item)}")

        # Optionally, you can print more details about the item
        if isinstance(item, h5py.Group):
            # If it's a group, explore it recursively
            explore_group(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            # If it's a dataset, print details about its shape and dtype
            print(" " * (indent + 2) + f"Dataset with shape: {item.shape}, dtype: {item.dtype}")

            # Optionally, you can perform actions on the dataset
            # For example, print a few values from the dataset
            print(" " * (indent + 2) + "Sample values:", item[:5])  # Print the first 5 values

# Open the HDF5 file
with h5py.File('data/cifar100vgg.h5', 'r') as file:
    # Assuming 'group_name' is the name of the group you want to explore
    group_name = 'conv2d_10'  # Replace with the actual group name
    if group_name in file:
        group = file[group_name]

        # Call the recursive function to explore the group
        print(f"Contents of the '{group_name}' group:")
        explore_group(group)
    else:
        print(f"Group '{group_name}' not found in the HDF5 file.")
