# Data

No data are bundled. Generate expert episodes with the README collection command, or obtain an agreed dataset from your group. Existing local datasets may live under /data/comp0245/runs on the original workstation; this repository intentionally does not follow those symlinks.

Preserve the manifest, scenario IDs, seeds, units, observation/action ordering, simulator settings and source hashes. Check expert completion and inspect failed episodes before training. Keep entire trajectories and related scenario families within one split; fit normalisation using training data only. Do not randomly split neighbouring time steps across train and evaluation.

Document any imported dataset's origin, permission, schema and adaptation. Do not assume another member's data have compatible action timing or units. Keep generated data outside Git and record a storage location and checksum for any dataset used in reported results.
