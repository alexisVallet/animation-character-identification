#pragma once

/**
 * Classifier computing one graph for each feature of interest in the
 * animation image - for instance one for position information, one for
 * color informatin, another for shape information. Characteristic vectors
 * are computed from each of these graphs, concatenated into long vectors
 * and classified using SVM.
 */
class MultipleGraphsClassifier {

};