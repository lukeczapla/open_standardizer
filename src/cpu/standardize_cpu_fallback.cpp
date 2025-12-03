#include <RDGeneral/RDLog.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolStandardize/MolStandardize.h>
#include <GraphMol/MolStandardize/Fragment.h>
#include <GraphMol/MolStandardize/Charge.h>
#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/MolStandardize/Normalize.h>
#include <GraphMol/MolStandardize/Metal.h>
#include <GraphMol/MolStandardize/Fragment.h>
#include <GraphMol/MolStandardize/MolVS.h>
#include <GraphMol/Descriptors/MolDescriptors.h>

#include "fallback_manager.hpp"
#include "cpu_utils.hpp"

// ===============================================
// CPU fallback pipeline
// Matches your ChemAxon XML sequence:
// 1. Clear stereo
// 2. Keep largest fragment
// 3. Remove atom values/data
// 4. Remove explicit hydrogens
// 5. Clear isotopes
// 6. Neutralize
// 7. Mesomerize (MolVS Normalization)
// 8. Tautomerize
// 9. Aromatize
// ===============================================

using namespace RDKit;

// ------------------------------------------------
// Utility: remove stereo flags
// ------------------------------------------------
static void clearStereo(ROMol &mol) {
    for (auto atom : mol.atoms()) {
        atom->setChiralTag(Atom::ChiralType::CHI_UNSPECIFIED);
    }
    for (auto bond : mol.bonds()) {
        bond->setStereo(Bond::BondStereo::STEREONONE);
    }
}

// ------------------------------------------------
// The real fallback pipeline
// ------------------------------------------------
std::unique_ptr<ROMol> cpu_fallback_standardize(const ROMol &inputMol) {
    RWMol *mol = new RWMol(inputMol);

    // -----------------------------------------------------------------------
    // (1) Clear stereochemistry (ChemAxon ClearStereo)
    // -----------------------------------------------------------------------
    clearStereo(*mol);

    // -----------------------------------------------------------------------
    // (2) Keep largest fragment
    // -----------------------------------------------------------------------
    {
        MolStandardize::LargestFragmentChooser chooser;
        ROMol *lm = chooser.pick(*mol);
        delete mol;
        mol = new RWMol(*lm);
        delete lm;
    }

    // -----------------------------------------------------------------------
    // (3) Normalize functional groups (ChemAxon Neutralize precursor)
    // -----------------------------------------------------------------------
    {
        MolStandardize::Normalizer normalizer;
        ROMol *nm = normalizer.normalize(*mol);
        delete mol;
        mol = new RWMol(*nm);
        delete nm;
    }

    // -----------------------------------------------------------------------
    // (4) Reionization / Neutralization
    // -----------------------------------------------------------------------
    {
        MolStandardize::Reionizer reionizer;
        ROMol *rm = reionizer.reionize(*mol);
        delete mol;
        mol = new RWMol(*rm);
        delete rm;
    }

    // -----------------------------------------------------------------------
    // (5) Remove explicit hydrogens
    // -----------------------------------------------------------------------
    {
        ROMol *hm = MolOps::removeHs(*mol, false, false);
        delete mol;
        mol = new RWMol(*hm);
        delete hm;
    }

    // -----------------------------------------------------------------------
    // (6) Clear isotopes
    // -----------------------------------------------------------------------
    for (auto atom : mol->atoms()) {
        atom->setIsotope(0);
    }

    // -----------------------------------------------------------------------
    // (7) Mesomerization / resonance normalization
    //     (MolVS Normalize covers ChemAxon Mesomerize normalization)
    // -----------------------------------------------------------------------
    {
        MolStandardize::Normalizer mesoNorm;
        ROMol *mm = mesoNorm.normalize(*mol);
        delete mol;
        mol = new RWMol(*mm);
        delete mm;
    }

    // -----------------------------------------------------------------------
    // (8) Tautomer canonicalization (ChemAxon Tautomerize)
    // -----------------------------------------------------------------------
    {
        MolStandardize::TautomerEnumerator te;
        ROMol *tm = te.canonicalize(*mol);
        delete mol;
        mol = new RWMol(*tm);
        delete tm;
    }

    // -----------------------------------------------------------------------
    // (9) Aromatize (ChemAxon Aromatize)
    // -----------------------------------------------------------------------
    MolOps::Kekulize(*mol, false);  // ensure kekule before aromaticity
    MolOps::sanitizeMol(*mol);      // includes aromaticity detection

    return std::unique_ptr<ROMol>(mol);
}
