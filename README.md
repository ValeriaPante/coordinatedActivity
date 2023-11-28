# Coordinated Activity on Twitter

In recent years, social networks have been leveraged in order to develop influence campaigns.
These deceptive and manipulative efforts seek to sway public opinion by disseminating fabricated and misleading information, usually with the intent of favoring a particular political side or ideology.
They take advantage of the discussion that naturally develops around such events to introduce misinformation and target common users.

Rather than being orchestrated by a solitary individual, influence campaigns tend to be sophisticated and coordinated endeavors undertaken by groups of malicious users operating in unison. These nefarious actors exploit a wide array of techniques and tactics.
There is an extended body of research that focuses on the detection of users belonging to influence campaigns.

This repository holds the implementation of state-of-the-art detection methods focused on building similarity networks of users based on different suspicious behavioural traces:

- Fast Retweet (Pacheco et al., 2020)
- Co-Retweet (Pacheco et al., 2021)
- Co-URL (Gabriel et al., 2023)
- Hashtag Sequence (Pacheco et al., 2021)
- Text Similarity (Pacheco et al., 2020)

Please cite the following work.

```
@misc{luceri2023unmasking,
      title={Unmasking the Web of Deceit: Uncovering Coordinated Activity to Expose Information Operations on Twitter},
      author={Luca Luceri and Valeria Pantè and Keith Burghardt and Emilio Ferrara},
      year={2023},
      eprint={2310.09884},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```

[Here](https://arxiv.org/abs/2310.09884) you can find the relative publication.

## Instructions

1. clone the repository using the command

`git clone https://github.com/ValeriaPante/coordinatedActivity.git -b ashwin`

2. To execute the detection methods, make use of the driver codes in the directory
   `scripts/INCAS_Drivers`

3. Steps for executing the detection methods
   - The drivers for the methods are mentioned below
     - coRetweet_driver.py
     - coURL_driver.py
     - hashtag_driver.py
     - text_similarity_driver.py
     - fastretweet_driver.py
   - Choose the method you wish to run and change the directories of the file according to your needs
     - `dataset_dir` = directory of where your dataset is stored
     - `graph_dir` = output directory for the network generated
     - `file_name` = name of the input file
   - Once changed, replicate the job file present in the `sample_jobs/sample_job.job` directory.
   - Change the jobfile based on the script you need to run. The default config is given in the jobfile.
   - Execute the jobfile with the command `sbatch sample_job.job`.

## Debug

- Incase of any issues backtrack to the source method files in the directory
  `scripts/coordinatedActivity/INCAS_scripts`
