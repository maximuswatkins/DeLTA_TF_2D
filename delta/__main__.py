"""
Entry point for the DeLTA executable standalone program.

This file is run by `python -m delta`, and typically reads the command-line
arguments given to this command (for example `python -m delta -i input.nd2
--bioformats 2D`).

It is also a good starting point to use DeLTA in a script.  In this case, you
can either simulate reading a command line, by passing a list of arguments to
`delta.cli.parse_args()`:

    args = delta.cli.parse_args(["-i", "input.nd2", "--bioformats", "2D"])

Or you can also remove the `args` line and replace the occurrences of
`args.<parameter>` by the value that you want.
"""

import delta

# Parse the command-line arguments
args = delta.cli.parse_args()

# Select the right configuration based on the specified presets
if args.presets == "2D":
    config = delta.config.DEFAULT_CONFIG_2D
elif args.presets == "mothermachine":
    config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE

# Create the file reader that reads the experiment file(s)
xpreader = delta.utils.XPReader(
    args.input,
    use_bioformats=args.bioformats,
    prototype=args.prototype,
    fileorder=args.order,
    filenamesindexing=args.indexing,
)

# Create the Pipeline that will orchestrate the running of DeLTA
xp = delta.pipeline.Pipeline(xpreader, config=config, resfolder=args.output)

# Run the Pipeline and save results
xp.process(positions=args.positions, nbframes=args.nbframes)

# Close the file reader
xpreader.close()
