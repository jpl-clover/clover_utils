import random

import clover_utils.generate_splits as generate_splits

def test_generate_10_random_splits():
    fp_src = f"./clover_utils/sample_data/msl/paper_provided_splits/train-set-v2.1.txt"

    # We want 10 different 'rolls'
    for i in range(10):
        # Change the seed, otherwise they'll all be the same
        print(f"Creating set of varying percentage of data files, with random seed {i}")
        random.seed(i)

        # Each has different fractions
        for pct in [1,5,10,20,30,40,50,60,70,80,90]:
            filenames, labels = generate_splits.read_filenames_and_labels(fp_src)
            new_filenames, new_labels = generate_splits.resample_fraction_with_preserved_distribution(filenames, labels, fraction=pct/100)
            generate_splits.write_filenames_and_labels(new_filenames, new_labels, f"./tests/outputs/random-split-{i}/{pct}pctTrain.txt")
            print(f"{pct}%-of-data regime has {len(new_filenames)} samples")

def test_generate_10_random_cumulative_splits():

    # To get a cumulative split, that is, the samples in the 10%-of-data regime are definitely 
    # contained in the 20%-of-data regime, we recursively sample from the previous regime. For example
    # we initially sample 90%-of-data from the 100%-of-data regime. Then we sample 80%-of-data from the
    # 90%-of-data regime. This ensures that the results are necessarily contained as we 'go down' the 
    # amount-of-data regimes. This ensures that samples are cumulative. It also ensures that the 
    # distribution of classes is preserved, provided that we use the 
    # 'resample_fraction_with_preserved_distribution' function

    # We want 10 different 'rolls'
    for i in range(10):
        # Change the seed, otherwise they'll all be the same
        print(f"Creating set of varying percentage of data files (cumulative), with random seed {i}")
        random.seed(i)

        # Each has different fractions. Reversed order is important
        pcts = [90,80,70,60,50,40,30,20,10,5,1]
        for j, pct in enumerate(pcts):
            # Where to find the previous X%-of-data regime file?
            if j == 0:
                fp_prev = f"./clover_utils/sample_data/msl/paper_provided_splits/train-set-v2.1.txt"
            else:
                fp_prev = f"./tests/outputs/random-cumulative-split-{i}/{pcts[j-1]}pctTrain.txt"
            fp_curr = f"./tests/outputs/random-cumulative-split-{i}/{pct}pctTrain.txt"

            # Load the prev files
            filenames, labels = generate_splits.read_filenames_and_labels(fp_prev)

            # If this is the first split, then just take 90%. If its the next split, we dont' want 80%, because it will be 80% of 
            # 90%. We want 80% of 100%. To do this, we need to take 80/90 of the 90% split (i.e. 88%)
            if j == 0:
                new_filenames, new_labels = generate_splits.resample_fraction_with_preserved_distribution(filenames, labels, fraction=pct/100)
            else:
                new_filenames, new_labels = generate_splits.resample_fraction_with_preserved_distribution(filenames, labels, fraction=pct/pcts[j-1])

            # Write the new files
            generate_splits.write_filenames_and_labels(new_filenames, new_labels, fp_curr)
            print(f"{pct}%-of-data regime has {len(new_filenames)} samples")

            # Lets double check that the cumulative splits really are cumulative
            all_contained = True
            for new_filename in new_filenames:
                all_contained = all_contained and (new_filename in filenames)
            print(f"Are all of the {pct}%\-of-data regime items contained in the {pcts[j-1]}%\-of-data regime? {all_contained}")


# TODO - check why the cumulative splits don't have the exact same amount as the regular random splits
# I would guess that because we need to preserve the dsitribution and the cumulative constraint it isn't
# always possible to select the right amount of files, hence why we're out +-10 per regime
test_generate_10_random_splits()
test_generate_10_random_cumulative_splits()
