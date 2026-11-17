const fs = require("fs"); let s = fs.readFileSync("content.js", "utf8");
function swap(startMarker, endMarker, repl) {
  const a = s.indexOf(startMarker), b = s.indexOf(endMarker);
  if (a < 0 || b < 0 || b < a) throw new Error("marker not found: " + startMarker.slice(0, 40));
  s = s.slice(0, a) + repl + s.slice(b);
}
const row = (n, p, f, sa, pl, v, d, vd) => `["${n}", "${p}", "${f}", "${sa}", "${pl}", "${d}", "${vd}"]`;
swap('add("h2", "4.3  First complete repeat', 'add("h2", "4.5  What the wavelets', `
add("h2", "4.3  Final repeated cross-validation (six repeats, 300 runs)");
add("p", "Table 2 gives the pooled out-of-fold results over three fold seeds × two training seeds, with the paired difference against the full model in each repeat and the all-repeats verdict. Three results are consistent in every repeat. (i) The interval-irregularity logistic regression is better than the full model (ΔAUROC +0.308 ± 0.030 on the best checkpoint; 6/0 repeats). (ii) The parameter-matched GRU twin is worse than the full model (−0.085 ± 0.076; 0/6). (iii) On the last-epoch checkpoint, the SNN without wavelets is better than the full model (+0.047 ± 0.042; 6/0), although on the best-validation-loss checkpoint the same comparison is not consistent (4/2). No other ablation differs consistently from the full model on either checkpoint, including fixing the wavelet filters (best: +0.020, 5/1), the unconstrained threshold, dead-band removal with decimation (lean), augmentation, spike dropout (best: +0.029, 5/1) and a hidden layer of 32 neurons (best: −0.030, 1/5). Between-repeat standard deviations of 0.03–0.07 are as large as most of these differences.");
add("table", "", { caption: "Table 2. Pooled out-of-fold performance over six repeats (mean ± SD). Δ = paired difference in window AUROC against full (best-validation-loss checkpoint), with the number of repeats in which the variant was better/worse. Verdicts use the all-six-repeats rule.",
  head: ["Variant", "Window AUROC (best)", "Subject AUROC (best)", "Window AUROC (last)", "Δ vs full (best)", "Better/worse", "Verdict (best | last)"],
  rows: [
    ["irreg_lr", "0.919 ± 0.003", "0.930 ± 0.007", "0.919 ± 0.003", "+0.308 ± 0.030", "6/0", "BETTER | BETTER"],
    ["snn_raw", "0.656 ± 0.057", "0.650 ± 0.069", "0.715 ± 0.034", "+0.046 ± 0.084", "4/2", "none | BETTER"],
    ["spkdrop", "0.640 ± 0.040", "0.641 ± 0.043", "0.694 ± 0.037", "+0.029 ± 0.026", "5/1", "none | none"],
    ["fixed_wavelet", "0.631 ± 0.044", "0.635 ± 0.039", "0.684 ± 0.045", "+0.020 ± 0.034", "5/1", "none | none"],
    ["aug", "0.631 ± 0.049", "0.631 ± 0.052", "0.685 ± 0.029", "+0.020 ± 0.044", "3/3", "none | none"],
    ["ht_despawn", "0.621 ± 0.045", "0.617 ± 0.044", "0.670 ± 0.035", "+0.010 ± 0.029", "3/3", "none | none"],
    ["lean", "0.613 ± 0.036", "0.617 ± 0.042", "0.650 ± 0.017", "+0.003 ± 0.047", "2/4", "none | none"],
    ["full", "0.611 ± 0.031", "0.615 ± 0.035", "0.668 ± 0.029", "—", "—", "—"],
    ["h32", "0.581 ± 0.055", "0.588 ± 0.056", "0.627 ± 0.045", "−0.030 ± 0.067", "1/5", "none | none"],
    ["gru_twin", "0.526 ± 0.066", "0.512 ± 0.076", "0.517 ± 0.047", "−0.085 ± 0.076", "0/6", "WORSE | WORSE"]],
  widths: [1250, 1300, 1300, 1300, 1400, 900, 1910] });
add("p", "(Wilcoxon p-values, shown for information in the results file, are 0.031 for the three consistent comparisons on the best checkpoint, the smallest possible with six pairs.)");
add("h2", "4.4  Statistical power");
add("p", "With six repeats the same-sign rule can only be satisfied by effects that exceed the run-to-run variation of 35 subjects; per-fold AUROC for the full model varied widely (for example 0.27–0.82 in the first repeat). The comparison with the interval baseline is the only large effect. Smaller differences, such as +0.02 to +0.03 for fixing the wavelet or adding spike dropout (5 of 6 repeats positive), neither confirm nor rule out a benefit.");
`);
swap('add("h2", "4.6  Edge export");', 'add("h1", "5  Discussion");', `add("h2", "4.6  Edge export");
add("p", "On the final models, all nine exportable configurations gave 100% prediction agreement between PyTorch and the NumPy implementation, with a maximum front-end difference of 1.2 × 10⁻⁶ and a maximum logit difference of 6.7 × 10⁻² (float32), and the exported full model file is 26.5 kB. On the workstation CPU used for the experiments (not the target device), removing the two dead bands and halving the number of time steps (lean) reduced per-window inference from 17 ms to 9 ms (−50%). [TBD: latency, CPU load, peak memory and power on the Raspberry Pi.] No energy figure is claimed.");
add("h2", "4.7  Secondary task (CinC 2015)");
add("p", "On the true-versus-false tachycardia-alarm task (383 windows: 166 true, 217 false alarms; 272 ventricular tachycardia, 111 tachycardia; one fold seed and one training seed), the full model and the interval baseline performed equally (pooled OOF AUROC 0.672 and 0.671; macro-F1 0.585 and 0.598). This is not AF detection and is a single repeat.");
`);
swap('add("h1", "5  Discussion");', 'add("h1", "6  Limitations");', `add("h1", "5  Discussion");
add("p", "The main finding is negative. On this cohort a learnable wavelet front end did not improve a small SNN, and the best neural model (the SNN on the raw signal, last checkpoint, AUROC 0.715) remained far below a three-feature classifier that measures the regularity of pulse intervals (0.919). That the baseline does well is unsurprising, since AF is defined by irregular rhythm and the features measure exactly that; it does set the bar any deep model must clear to justify its cost on this task.");
add("p", "Three observations qualify the picture. First, the parameter-matched GRU twin was worse than the SNN in every repeat, so the spiking recurrent core is not what limits performance relative to a non-spiking model of the same size. Second, the wavelet stage did not help: the model without wavelets was better on the last checkpoint in every repeat, and fixing the filters was no worse than learning them. Third, in the one fold inspected the learned thresholds and encoder parameters stayed close to their initial values (Section 4.5), and validation AUROC (about 0.98) was much higher than test AUROC on held-out subjects, which points to overfitting to subject-specific features in a 35-subject cohort. Candidate explanations that we did not test and do not assert are that the db4 sub-band split removes beat-to-beat timing information that the raw input retains, and that the regularisers hold the front end near its initialisation.");
add("p", "The engineering contributions stand independently of the accuracy outcome: the closed-form readout, leak rescaling under decimation (which halved latency without a consistent loss in accuracy), the shrink-only threshold, the deterministic protocol and the exact NumPy export. The evaluation protocol also changed conclusions: in the earlier version the same configuration produced AUROC 0.965 and 0.707 in two runs, and a single repeat of the present experiment would have ranked some neural variants differently from the six-repeat verdicts. Larger open PPG datasets with AF labels would be needed to test whether a learnable wavelet front end helps at all, and we provide the code for that purpose.");
`);
swap('add("h1", "7  Conclusion");', 'add("h1", "Declarations");', `add("h1", "7  Conclusion");
add("p", "We presented WaveSpike-PPG, a 5,200-parameter spiking network with a learnable causal wavelet bank, a shrink-only learnable threshold and a sign-preserving two-channel spike encoder, together with a leakage-controlled repeated evaluation protocol and an exact NumPy export. In 300 runs on 35 subjects, no variant of the wavelet-SNN beat a three-feature interval-irregularity baseline, which was better in every repeat, and the wavelet front end gave no consistent gain over fixed filters or over an SNN without wavelets. We therefore report a negative result and a reproducible reference implementation, and leave open whether learnable wavelets help on larger and more varied PPG cohorts, and how the exported model behaves on a Raspberry Pi.");
`);
fs.writeFileSync("content.js", s); console.log("patched");
