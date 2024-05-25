# Model Results

This folder contains a collection of pre-scanned laser results for various models. If you scan a different model and find results that are not already included here, we encourage you to submit a Pull Request (PR) to add those results. We will continue to update this repository as we use Laser in our Dolphin models.

## Usage

To use laser scanner with any of these results, run the following command:

```bash
python laser_scanner.py --json <path to snr-results.json> --top_percent <the top % of dense layers you want to target>
```

Replace `<path to snr-results.json>` with the path to the SNR results JSON file and `<the top % of dense layers you want to target>` with the percentage of the top dense layers you wish to target.

## Contributing

We welcome contributions! If you would like to contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new laser results for model XYZ'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please make sure your contributions include:
- The JSON file containing the SNR results.
- A link to the Hugging Face (HF) repository the results came from.
