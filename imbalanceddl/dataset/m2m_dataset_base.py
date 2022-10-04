
class M2mBaseDataset:
    """Base Dataset (Mixin)

    Base dataset for creating imbalanced dataset
    """
    def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
        """
        Return a list of imbalanced indices from a dataset.
        Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
        Output: oversampled_list ( weights are increased )
        """
        length = dataset.__len__()
        num_sample_per_class = list(num_sample_per_class)
        num_samples = list(num_sample_per_class)

        selected_list = []
        indices = list(range(0,length))
        for i in range(0, length):
            index = indices[i]
            _, label = dataset.__getitem__(index)
            if num_sample_per_class[label] > 0:
                selected_list.append(1 / num_samples[label])
                num_sample_per_class[label] -= 1
        print(len(selected_list))
        return selected_list

    def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
        """
        Return a list of imbalanced indices from a dataset.
        Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
        Output: imbalanced_list
        """
        length = dataset.__len__()
        num_sample_per_class = list(num_sample_per_class)
        selected_list = []
        indices = list(range(0,length))

        for i in range(0, length):
            index = indices[i]
            _, label = dataset.__getitem__(index)
            if num_sample_per_class[label] > 0:
                selected_list.append(index)
                num_sample_per_class[label] -= 1

        return selected_list

