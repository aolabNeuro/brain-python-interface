/**
 * Vue.js based Features Component
 * Manages feature selection and toggling
 * 
 * Replaces jQuery-based feature checkbox handling
 */

const featuresApp = {
    el: '#features_vue',
    data() {
        return {
            features: [],  // Will be populated from template data
            selectedFeatures: {},  // Map of feature name -> selected boolean
            searchQuery: '',  // For filtering features
        };
    },
    computed: {
        /**
         * Filter features based on search query
         */
        filteredFeatures() {
            if (!this.searchQuery) {
                return this.features;
            }
            const query = this.searchQuery.toLowerCase();
            return this.features.filter(f => 
                f.name.toLowerCase().includes(query) ||
                f.desc.toLowerCase().includes(query)
            );
        },
        /**
         * Count selected features
         */
        selectedCount() {
            return Object.values(this.selectedFeatures).filter(Boolean).length;
        },
        /**
         * Get array of selected feature IDs for form submission
         */
        selectedFeatureIds() {
            return this.features
                .filter(f => this.selectedFeatures[f.name])
                .map(f => f.id);
        }
    },
    methods: {
        /**
         * Toggle a feature's selected state
         */
        toggleFeature(featureName) {
            this.selectedFeatures[featureName] = !this.selectedFeatures[featureName];
        },

        /**
         * Initialize features from the template data
         */
        initializeFeatures(featuresList) {
            this.features = featuresList;
            // Initialize selectedFeatures object
            const selectedObj = {};
            featuresList.forEach(f => {
                selectedObj[f.name] = false;  // Start with all unchecked
            });
            this.selectedFeatures = selectedObj;
        },

        /**
         * Select all features
         */
        selectAll() {
            this.filteredFeatures.forEach(f => {
                this.selectedFeatures[f.name] = true;
            });
        },

        /**
         * Deselect all features
         */
        deselectAll() {
            this.filteredFeatures.forEach(f => {
                this.selectedFeatures[f.name] = false;
            });
        },

        /**
         * Get selected features for submission to server
         */
        getSelectedFeatureNames() {
            return Object.keys(this.selectedFeatures)
                .filter(name => this.selectedFeatures[name]);
        },

        /**
         * Set selected features (called from jQuery code)
         */
        setSelectedFeatures(featureNames) {
            const selected = {};
            this.features.forEach(f => {
                // Handle both array and object formats
                if (Array.isArray(featureNames)) {
                    selected[f.name] = featureNames.includes(f.name);
                } else if (typeof featureNames === 'object' && featureNames !== null) {
                    // If it's an object, check if the key exists and is truthy
                    selected[f.name] = !!featureNames[f.name];
                } else {
                    selected[f.name] = false;
                }
            });
            this.selectedFeatures = selected;
        }
    },

    mounted() {
        console.log("Features Vue app mounted");
    }
};

// Export for use in list.js
window.featuresApp = featuresApp;

/**
 * Bridge function to initialize features from template data
 * Called from list.js when TaskEntry is created
 */
window.initializeVueFeatures = function(featuresList) {
    if (featuresApp.instance && featuresApp.instance.initializeFeatures) {
        featuresApp.instance.initializeFeatures(featuresList);
    }
};

/**
 * Bridge function to get selected features from Vue
 * Called from list.js when submitting form
 */
window.getVueSelectedFeatures = function() {
    if (featuresApp.instance) {
        return featuresApp.instance.getSelectedFeatureNames();
    }
    return [];
};
