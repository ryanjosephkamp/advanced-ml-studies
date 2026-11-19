# Data sources

The EEG data this project uses come from three public datasets. This repository's MIT License does not cover them; each keeps its own terms.

`scripts/download_data.py` downloads the raw files the code reads from the sources below and checks each against `raw_files.sha256` (408 files: EEG Motor Movement/Imagery runs 4, 8, and 12 of all 109 subjects; the 72 EEGMAT recordings; BCI Competition IV 2a training sessions A01T to A09T). The PhysioNet checksums are PhysioNet's published SHA256SUMS; BCI Competition IV publishes none, so those 9 are the sums of the files in its zip as downloaded on November 11, 2026 (439,968,864 bytes). The script lays the files out as `eeg_mmi/`, `eegmat/`, and `bci_iv_2a/` under the folder you give it, which is the folder to pass as `--raw-dir`.

## EEG Motor Movement/Imagery Dataset (`eeg_mmi/`)

- Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., and Wolpaw, J. R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.
- License: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

## EEG During Mental Arithmetic Tasks (`eeg_ma/`)

- Zyma, I., Tukaev, S., Seleznov, I., Kiyono, K., Popov, A., Chernykh, M., and Shpenkov, O. (2019). Electroencephalograms during Mental Arithmetic Task Performance. *Data*, 4(1), 14. https://doi.org/10.3390/data4010014
- Dataset: [PhysioNet EEGMAT](https://physionet.org/content/eegmat/1.0.0/) (version 1.0.0). https://doi.org/10.13026/C2JQ1P
- License: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

## BCI Competition IV, data set 2a (`bci_civ_2a/`)

- Recorded at Graz University of Technology. Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., and Pfurtscheller, G. (2008). [BCI Competition 2008 – Graz data set A](https://www.bbci.de/competition/iv/desc_2a.pdf).
- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55. https://doi.org/10.3389/fnins.2012.00055
- Terms: the [competition](https://www.bbci.de/competition/iv/) asks users to credit the group that recorded the data and to cite a paper from the data set's description.

## PhysioNet

PhysioNet asks users of its data to include its standard citation: Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., and Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z
