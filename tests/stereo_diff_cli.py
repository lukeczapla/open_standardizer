"""
stereo_diff_cli.py

Command-line tool to compare ChemAxon-style enhanced stereo annotations
in curly blocks (A*/B*/groups) against RDKit's CIP assignments.

Input
-----
CSV with either:
    1) id,smiles
    2) smiles

Where `smiles` may be plain SMILES or ChemAxon-enhanced SMILES, e.g.:

    12345,CCN(CC)CC {A7=s;A9=r;o1:7,9}
    "C/C=C\\C {B0,1=e}"
    CCO
    67890,"c1ccccc1[NH2+] {A20=p;B5,6=e;&1:7;&2:20}"

Quotes are handled by the CSV reader; commas inside SMILES are fine.

Behavior
--------
For each SMILES with a curly block, the tool:

  - Parses the block via enhanced_smiles.py (A*, B*, o/& groups).
  - Builds an RDKit Mol from the core SMILES.
  - Uses stereo_validation.validate_cip_assignments_on_smiles(...)
    to compare:

      * Atom-level A<idx>=R/S vs RDKit `_CIPCode` on atoms.
      * Bond-level B<i,j>=E/Z vs RDKit `_CIPCode` on bonds.

  - Non-CIP atom codes (e.g. A20=p) are recorded but *not* treated
    as mismatches; they are reported as "non_cip_code".

The tool summarizes counts and can optionally write a detailed issues
CSV for downstream analysis.

Usage
-----
    python -m chemstandardizer.stereo_diff_cli input.csv
    python -m chemstandardizer.stereo_diff_cli input.csv --report issues.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import Optional

from ..stereo_validation import validate_cip_assignments_on_smiles


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ChemAxon-style enhanced stereo (A*/B* in { ... }) "
            "against RDKit CIP assignments over a CSV of SMILES."
        )
    )
    parser.add_argument(
        "input_csv",
        help=(
            "Input CSV file with either 'id,smiles' or 'smiles' rows. "
            "SMILES may include ChemAxon-style curly blocks."
        ),
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help=(
            "Optional output CSV for per-record stereo issues. "
            "If omitted, only a summary is printed."
        ),
    )

    args = parser.parse_args(argv)

    total_rows = 0
    rows_with_meta = 0

    atom_matches = atom_mismatches = 0
    atom_missing = atom_non_cip = 0

    bond_matches = bond_mismatches = 0
    bond_missing = bond_unknown = 0

    report_writer = None
    report_fh = None

    if args.report:
        report_fh = open(args.report, "w", newline="")
        report_writer = csv.writer(report_fh)
        report_writer.writerow(
            [
                "id",
                "smiles",
                "level",        # "atom" or "bond"
                "index1",       # atom index or first atom index
                "index2",       # second atom index (for bonds) or empty
                "expected",     # ChemAxon annotation (A/B code)
                "cip",          # RDKit CIP code, if present
                "status",       # match/mismatch/missing_cip/non_cip_code/...
            ]
        )

    with open(args.input_csv, newline="") as fh:
        reader = csv.reader(fh)

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            total_rows += 1

            # Accept either: [smiles] OR [id,smiles,...]
            if len(row) == 1:
                rec_id = str(line_number)
                smiles = row[0].strip()
            else:
                rec_id = row[0].strip()
                smiles = row[1].strip()

            if not smiles:
                continue

            # index_base is always 0 here: A0, B0,1, etc.
            res = validate_cip_assignments_on_smiles(smiles, index_base=0)
            if res is None:
                # Either no curly metadata or parse failure; skip
                continue

            rows_with_meta += 1

            # Aggregate atom stats
            atom_matches      += len(res.atom_matches)
            atom_mismatches   += len(res.atom_mismatches)
            atom_missing      += len(res.atom_missing_cip)
            atom_non_cip      += len(res.atom_non_cip_assignments)

            # Aggregate bond stats
            bond_matches      += len(res.bond_matches)
            bond_mismatches   += len(res.bond_mismatches)
            bond_missing      += len(res.bond_missing_cip)
            bond_unknown      += len(res.bond_unknown_assignments)

            if report_writer:
                # Atoms: mismatches, missing CIP, and non-CIP codes
                for a in (
                    res.atom_mismatches
                    + res.atom_missing_cip
                    + res.atom_non_cip_assignments
                ):
                    report_writer.writerow(
                        [
                            rec_id,
                            smiles,
                            "atom",
                            a.atom_index,
                            "",
                            a.assignment,
                            a.cip_code or "",
                            a.status,
                        ]
                    )

                # Bonds: mismatches, missing CIP, and unknown expected codes
                for b in (
                    res.bond_mismatches
                    + res.bond_missing_cip
                    + res.bond_unknown_assignments
                ):
                    report_writer.writerow(
                        [
                            rec_id,
                            smiles,
                            "bond",
                            b.atom_index1,
                            b.atom_index2,
                            b.assignment,
                            b.cip_code or "",
                            b.status,
                        ]
                    )

    if report_fh:
        report_fh.close()

    # Summary
    print(f"Input rows processed:        {total_rows}")
    print(f"Rows with curly metadata:    {rows_with_meta}")
    print()
    print("Atom-level CIP (A<idx>=...):")
    print(f"  matches:                   {atom_matches}")
    print(f"  mismatches:                {atom_mismatches}")
    print(f"  missing CIP on RDKit side: {atom_missing}")
    print(f"  non-CIP atom codes (e.g. p): {atom_non_cip}")
    print()
    print("Bond-level CIP (B<i,j>=...):")
    print(f"  matches:                   {bond_matches}")
    print(f"  mismatches:                {bond_mismatches}")
    print(f"  missing CIP on RDKit side: {bond_missing}")
    print(f"  unknown/other bond codes:  {bond_unknown}")
    if args.report:
        print()
        print(f"Detailed issues written to:  {args.report}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
