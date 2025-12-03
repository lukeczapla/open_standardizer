#include "gpu_kernels.hpp"
#include "stereo_kernels.hpp"

#include <cuda_runtime.h>
#include <RDGeneral/Exceptions.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>

// Convert RDKit molecule into GPU buffers (simplified)
static void mol_to_gpu_arrays(
    const RDKit::ROMol &mol,
    std::vector<GAtom> &atoms,
    std::vector<GBond> &bonds
) {
    atoms.reserve(mol.getNumAtoms());
    bonds.reserve(mol.getNumBonds());

    for (auto atom : mol.atoms()) {
        GAtom a;
        a.atomicNum = atom->getAtomicNum();
        a.hCount = atom->getTotalNumHs();

        if (atom->getChiralTag() == RDKit::Atom::CHI_TETRAHEDRAL_CW)
            a.chiralFlag = 1;
        else if (atom->getChiralTag() == RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            a.chiralFlag = 2;
        else
            a.chiralFlag = 3;

        atoms.push_back(a);
    }

    for (auto bond : mol.bonds()) {
        GBond b;
        b.order = bond->getBondTypeAsDouble();
        b.idxA = bond->getBeginAtomIdx();
        b.idxB = bond->getEndAtomIdx();

        auto st = bond->getStereo();
        switch (st) {
        case RDKit::Bond::STEREOZ: b.stereo = 4; break;
        case RDKit::Bond::STEREOE: b.stereo = 3; break;
        case RDKit::Bond::STEREOANY:
        case RDKit::Bond::STEREONONE:
        case RDKit::Bond::STEREOZORTRANS:
        case RDKit::Bond::STEREOCTORTRANS:
        default:
            b.stereo = 0;
        }

        bonds.push_back(b);
    }
}

// Apply GPU results → RDKit molecule
static void apply_stereo_back(
    RDKit::ROMol &mol,
    const std::vector<StereoFixResult> &atoms,
    const std::vector<StereoFixResult> &bonds
) {
    // atoms
    for (size_t i = 0; i < atoms.size(); i++) {
        const auto &r = atoms[i];
        auto a = mol.getAtomWithIdx(i);

        switch (r.atomChiral) {
        case 1: a->setChiralTag(RDKit::Atom::CHI_TETRAHEDRAL_CW);  break;
        case 2: a->setChiralTag(RDKit::Atom::CHI_TETRAHEDRAL_CCW); break;
        default: a->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);    break;
        }
    }

    // bonds
    for (size_t i = 0; i < bonds.size(); i++) {
        const auto &r = bonds[i];
        auto b = mol.getBondWithIdx(i);

        switch (r.bondStereo) {
        case 3: b->setStereo(RDKit::Bond::STEREOE); break;
        case 4: b->setStereo(RDKit::Bond::STEREOZ); break;
        default: b->setStereo(RDKit::Bond::STEREONONE); break;
        }
    }
}


// ============================================================
// PUBLIC ENTRYPOINT
// ============================================================

RDKit::ROMol gpu_kernel_stereo(const RDKit::ROMol &inMol) {
    RDKit::ROMol mol(inMol);

    // CPU → GPU transfer
    std::vector<GAtom> hAtoms;
    std::vector<GBond> hBonds;
    mol_to_gpu_arrays(mol, hAtoms, hBonds);

    int nA = hAtoms.size();
    int nB = hBonds.size();

    GAtom *dAtoms = nullptr;
    GBond *dBonds = nullptr;
    StereoFixResult *dAOut = nullptr;
    StereoFixResult *dBOut = nullptr;

    cudaMalloc(&dAtoms, nA * sizeof(GAtom));
    cudaMalloc(&dBonds, nB * sizeof(GBond));
    cudaMalloc(&dAOut, nA * sizeof(StereoFixResult));
    cudaMalloc(&dBOut, nB * sizeof(StereoFixResult));

    cudaMemcpy(dAtoms, hAtoms.data(), nA * sizeof(GAtom), cudaMemcpyHostToDevice);
    cudaMemcpy(dBonds, hBonds.data(), nB * sizeof(GBond), cudaMemcpyHostToDevice);

    // Launch kernel
    int block = 128;
    int grid = (max(nA, nB) + block - 1) / block;

    stereo_kernel<<<grid, block>>>(
        dAtoms, dBonds, dAOut, dBOut, nA, nB
    );
    cudaDeviceSynchronize();

    // Collect results
    std::vector<StereoFixResult> outA(nA), outB(nB);

    cudaMemcpy(outA.data(), dAOut, nA * sizeof(StereoFixResult), cudaMemcpyDeviceToHost);
    cudaMemcpy(outB.data(), dBOut, nB * sizeof(StereoFixResult), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(dAtoms);
    cudaFree(dBonds);
    cudaFree(dAOut);
    cudaFree(dBOut);

    apply_stereo_back(mol, outA, outB);

    return mol;
}
