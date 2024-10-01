We will use Cat Meow Classification dataset, which consist of 21 cats' meow sounds. This dataset, composed of 440 sound recordings, contains meows emitted by cats in different contexts. Specifically, 21 cats belonging to 2 breeds (Maine Coon and European Shorthair) have been repeatedly exposed to three different stimuli that act as labels for prediction:

Waiting for food;
Isolation in unfamiliar environment;
Brushing (being brushed affectionately by the owner).
Files containing meows are in the "dataset" folder. They are PCM streams (.wav). Naming conventions follow the pattern L[label]_CID[Cat ID]_BB[Cat Breed]_SS[Sex]_OID[Owner ID]_R[Recording Session]XX[Meow counter]. The unique values are available in the 'dataset' folder description.

Data directory.
Naming convention for files -> C_NNNNN_BB_SS_OOOOO_RXX, where:

C = emission context (values: B = brushing; F = waiting for food; I: isolation in an unfamiliar environment);
NNNNN = cat’s unique ID;
BB = breed (values: MC = Maine Coon; EU: European Shorthair);
SS = sex (values: FI = female, intact; FN: female, neutered; MI: male, intact; MN: male, neutered);
OOOOO = cat owner’s unique ID;
R = recording session (values: 1, 2 or 3)
XX = vocalization counter (values: 01..99)