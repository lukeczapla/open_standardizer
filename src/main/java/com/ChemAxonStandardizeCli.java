package com.example;

import chemaxon.formats.MolImporter;
import chemaxon.formats.MolExporter;
import chemaxon.standardizer.Standardizer;
import chemaxon.struc.Molecule;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal ChemAxon Standardizer CLI.
 *
 * Usage:
 *   java -cp "chemaxon.jar:." com.example.ChemAxonStandardizeCli \
 *        --xml standardizer.xml \
 *        --in data.csv \
 *        --out-format smiles
 *
 * Comparison Usage:
 * python -m open_standardizer.tests.standardize_cli \
 *   --xml standardizer.xml \
 *   --in library.csv \
 *   --out-format enhanced > open_std_out.tsv
 * 
 * 
 * If --in is omitted, reads CSV from stdin.
 * CSV rules:
 *   - If row has 1 column: treated as (id=smiles, smiles)
 *   - If row has >=2: (id=row[0], smiles=row[1])
 *
 * Output: TSV to stdout:
 *   id<TAB>status<TAB>input_smiles<TAB>standardized
 *
 * status = OK or ERROR:<reason>
 */
public class ChemAxonStandardizeCli {

    private static class Args {
        String xmlPath;
        String inputPath = null;
        String outFormat = "smiles";   // "smiles" or "cxsmiles"
    }

    private static void usage() {
        System.err.println("ChemAxon Standardizer CLI");
        System.err.println();
        System.err.println("Required:");
        System.err.println("  --xml <standardizer.xml>");
        System.err.println();
        System.err.println("Optional:");
        System.err.println("  --in <file.csv>     (if omitted, read from stdin)");
        System.err.println("  --out-format smiles | cxsmiles   (default: smiles)");
        System.err.println();
        System.err.println("CSV rules:");
        System.err.println("  - 1-col row:  id = smiles, smiles = col0");
        System.err.println("  - >=2-col row: id = col0, smiles = col1");
        System.err.println();
    }

    private static Args parseArgs(String[] argv) {
        Args args = new Args();
        for (int i = 0; i < argv.length; ++i) {
            String a = argv[i];
            if (a.equals("--xml") && i + 1 < argv.length) {
                args.xmlPath = argv[++i];
            } else if (a.equals("--in") && i + 1 < argv.length) {
                args.inputPath = argv[++i];
            } else if (a.equals("--out-format") && i + 1 < argv.length) {
                args.outFormat = argv[++i].toLowerCase();
            } else if (a.equals("-h") || a.equals("--help")) {
                usage();
                System.exit(0);
            }
        }
        if (args.xmlPath == null) {
            usage();
            throw new IllegalArgumentException("Missing --xml <standardizer.xml>");
        }
        if (!args.outFormat.equals("smiles") && !args.outFormat.equals("cxsmiles")) {
            throw new IllegalArgumentException("Unsupported --out-format: " + args.outFormat);
        }
        return args;
    }

    private static BufferedReader openReader(String path) throws IOException {
        if (path == null) {
            return new BufferedReader(new InputStreamReader(System.in));
        }
        return new BufferedReader(new FileReader(path));
    }

    // Simple CSV splitter (no full RFC handling, but fine for id,smiles).
    private static List<String> splitCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); ++i) {
            char c = line.charAt(i);
            if (c == '"') {
                // toggle quote mode; handle doubled quotes if you want to be fancy
                inQuotes = !inQuotes;
                continue;
            }
            if (c == ',' && !inQuotes) {
                out.add(sb.toString());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        out.add(sb.toString());
        return out;
    }

    private static String standardizeOne(
            Standardizer std,
            String smiles,
            String outFormat
    ) throws Exception {
        // Import as SMILES (ChemAxon will also parse CXSMILES .. but here we assume plain / CX)
        Molecule mol = MolImporter.importMol(smiles, "smiles");
        if (mol == null) {
            throw new IllegalArgumentException("Invalid SMILES: " + smiles);
        }

        std.standardize(mol);

        if (outFormat.equals("smiles")) {
            return MolExporter.exportToFormat(mol, "smiles");
        } else {
            // CXSMILES with default features (e, l, w, d, f, p, R, L, m, N, D).
            // You can add ":+c" to always include coords, etc.
            return MolExporter.exportToFormat(mol, "cxsmiles:");
        }
    }

    public static void main(String[] argv) throws Exception {
        Args args;
        try {
            args = parseArgs(argv);
        } catch (IllegalArgumentException ex) {
            System.err.println("ERROR: " + ex.getMessage());
            System.exit(2);
            return;
        }

        // Load Standardizer from XML
        Standardizer std = new Standardizer(new FileInputStream(args.xmlPath));

        try (BufferedReader in = openReader(args.inputPath)) {
            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                List<String> cols = splitCsvLine(line);
                if (cols.isEmpty()) {
                    continue;
                }
                String id;
                String smiles;
                if (cols.size() == 1) {
                    smiles = cols.get(0).trim();
                    id = smiles;
                } else {
                    id = cols.get(0).trim();
                    smiles = cols.get(1).trim();
                    if (id.isEmpty()) {
                        id = smiles;
                    }
                }

                if (smiles.isEmpty()) {
                    continue;
                }

                try {
                    String out = standardizeOne(std, smiles, args.outFormat);
                    System.out.println(id + "\tOK\t" + smiles + "\t" + out);
                } catch (Exception ex) {
                    System.out.println(id + "\tERROR:" + ex.getMessage() + "\t" + smiles + "\t");
                }
            }
        }
    }
}
