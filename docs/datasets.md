# Dataset Notes

This repository keeps raw dataset files in dedicated local folders and excludes them from GitHub commits.

## Runtime Files

The current application reads these files at runtime:

- `Occupation Data.xlsx`
- `Skills.xlsx`
- `Knowledge.xlsx`
- `Work Activities.xlsx`
- `Interests.xlsx`
- `Work Styles.xlsx`
- `Technology Skills.xlsx`
- `Job Zones.xlsx`
- `Job Zone Reference.xlsx`
- `Related Occupations.xlsx`
- `Career Dataset.xlsx`
- `all_job_post.csv`

They are now organized under:

- `data/runtime/`

If you want to run the app after this cleanup, point the project there with:

- `STRUX_DATA_ROOT=data/runtime`

You can also point the app to another dataset location with:

- `STRUX_DATA_ROOT`

## Reference Files

Other O*NET export files currently present in the local workspace are not used directly by the running application.
They can remain local as archival/reference material and are intentionally not part of the GitHub upload set.

They are organized under:

- `data/reference/onet/`

## Why They Are Ignored

- Raw source files are large and make the repository noisy.
- Some files are licensed or distributed separately from the application code.
- The codebase is easier to review when source code, config, examples, and tests are separated from bulk raw data.
