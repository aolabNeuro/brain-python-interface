/**
 * Vue.js based Parameters Component
 * Manages experiment parameters display and editing
 * 
 * Replaces jQuery-based parameter table rendering with reactive Vue
 */

const parametersApp = {
    el: '#parameters_vue',
    data() {
        return {
            parameters: {},  // Map of param_name -> {value, type, desc, options, inputs, etc}
            showAllParams: false,
            filteredParams: {},
            editMode: false,
        };
    },
    computed: {
        /**
         * Get visible parameters based on showAllParams flag
         */
        visibleParams() {
            if (this.showAllParams) {
                return this.parameters;
            }
            // Show only non-hidden parameters
            // Parameters with hidden='hidden' should be hidden unless showAllParams is true
            return Object.fromEntries(
                Object.entries(this.parameters).filter(([name, param]) => {
                    return param.hidden !== 'hidden';
                })
            );
        },

        /**
         * Count of parameters
         */
        paramCount() {
            return Object.keys(this.parameters).length;
        },

        /**
         * Count of visible parameters
         */
        visibleCount() {
            return Object.keys(this.visibleParams).length;
        }
    },

    methods: {
        /**
         * Update parameters from descriptor object
         * Format: {param_name: {value, type, desc, options, default, ...}}
         * 
         * The descriptor comes from the Django backend Task.params() method
         * and contains all the metadata needed to display parameter inputs
         */
        updateParameters(desc) {
            // Initialize parameters with provided descriptor
            // If no value is set, we'll use the default when capturing
            // but still keep the value undefined for now so placeholder shows
            const params = {};
            for (let name in desc) {
                params[name] = { ...desc[name] };
                // Ensure value is set to something (use default if not provided)
                if (params[name].value === undefined || params[name].value === null) {
                    if (params[name].type === 'Bool') {
                        params[name].value = params[name].default || false;
                    } else if (['Float', 'Int'].includes(params[name].type)) {
                        params[name].value = params[name].default || null;
                    } else if (Array.isArray(params[name].default)) {
                        // Tuple, Array, List types
                        params[name].value = Array.isArray(params[name].value) ? 
                            params[name].value : 
                            (params[name].default || []);
                    } else {
                        params[name].value = params[name].default || null;
                    }
                }
            }
            
            this.parameters = params;
            this.showAllParams = false;  // Reset to show only visible parameters
        },

        /**
         * Toggle visibility of all parameters
         */
        toggleShowAll() {
            this.showAllParams = !this.showAllParams;
        },

        /**
         * Get parameter by name
         */
        getParameter(name) {
            return this.parameters[name];
        },

        /**
         * Set parameter value
         */
        setParameterValue(name, value) {
            if (this.parameters[name]) {
                this.parameters[name].value = value;
            }
        },

        /**
         * Get all parameter values as object
         */
        getParameterValues() {
            const values = {};
            for (let name in this.parameters) {
                values[name] = this.parameters[name].value;
            }
            return values;
        },

        /**
         * Format parameter value for display
         */
        formatValue(param) {
            if (param.value === null || param.value === undefined) {
                return '';
            }
            if (Array.isArray(param.value)) {
                return param.value.join(', ');
            }
            return String(param.value);
        },

        /**
         * Check if parameter is of given type
         */
        isType(param, type) {
            return param.type === type;
        },

        /**
         * Get option key for v-for :key binding
         * Handles both tuple format [pk, path] and simple string values
         */
        getOptionKey(option) {
            if (Array.isArray(option)) {
                return option[0];  // Use first element (pk) as key
            }
            return option;
        },

        /**
         * Get option value for v-model binding
         * Handles both tuple format [pk, path] and simple string values
         */
        getOptionValue(option) {
            if (Array.isArray(option)) {
                return option[0];  // Use first element (pk) as value
            }
            return option;
        },

        /**
         * Get option display text
         * For tuples [pk, path], displays path (second element)
         * For simple strings, displays the string itself
         */
        getOptionText(option) {
            if (Array.isArray(option)) {
                return option[1];  // Use second element (path/display) for text
            }
            return option;
        },

        /**
         * Enable/disable editing
         */
        setEditMode(enabled) {
            this.editMode = enabled;
        }
    },

    mounted() {
        console.log("Parameters Vue app mounted");
    }
};

// Export for use in list.js
window.parametersApp = parametersApp;

/**
 * Bridge function to update parameters from jQuery code
 * Called when task parameters are loaded/updated
 */
window.updateVueParameters = function(parametersDesc) {
    if (parametersApp.instance && parametersApp.instance.updateParameters) {
        parametersApp.instance.updateParameters(parametersDesc);
    }
};

/**
 * Bridge function to get parameter values from Vue
 * Called when submitting form
 */
window.getVueParameterValues = function() {
    if (parametersApp.instance) {
        return parametersApp.instance.getParameterValues();
    }
    return {};
};

/**
 * Bridge function to enable/disable parameter editing
 */
window.setVueParametersEditMode = function(enabled) {
    if (parametersApp.instance && parametersApp.instance.setEditMode) {
        parametersApp.instance.setEditMode(enabled);
    }
};
