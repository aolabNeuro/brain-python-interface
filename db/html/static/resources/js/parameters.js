/**
 * Vue.js based Parameters Component
 * Manages experiment parameters display and editing
 * 
 * Replaces jQuery-based parameter table rendering with reactive Vue
 */

const parametersRoot = typeof window !== 'undefined' ? window : globalThis;

if (typeof(require) !== 'undefined' && typeof($) === 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(``);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}

const parametersApp = {
    el: '#parameters_vue',
    data() {
        return {
            parameters: {},  // Map of param_name -> {value, type, desc, options, inputs, etc}
            showAllParams: false,
            searchQuery: '',
            sortMode: 'default',
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
            // Show only non-hidden parameters unless showAllParams is enabled
            return Object.fromEntries(
                Object.entries(this.parameters).filter(([name, param]) => {
                    return !this.isParamHidden(param);
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
        },

        /**
         * Visible parameters filtered by search query
         */
        filteredVisibleParams() {
            if (!this.searchQuery) {
                return this.visibleParams;
            }

            const query = this.searchQuery.toLowerCase();
            return Object.fromEntries(
                Object.entries(this.visibleParams).filter(([name, param]) => {
                    const label = String(param.label || name || '').toLowerCase();
                    const desc = String(param.desc || '').toLowerCase();
                    return label.includes(query) || desc.includes(query);
                })
            );
        },

        /**
         * Count of visible parameters after search filter
         */
        filteredVisibleCount() {
            return Object.keys(this.filteredVisibleParams).length;
        },

        /**
         * Filtered parameters as a sorted array for stable rendering order control
         */
        sortedFilteredVisibleParams() {
            const entries = Object.entries(this.filteredVisibleParams);

            if (this.sortMode === 'default') {
                return entries;
            }

            const sorted = entries.sort(([nameA, paramA], [nameB, paramB]) => {
                const labelA = String(paramA?.label || nameA || '');
                const labelB = String(paramB?.label || nameB || '');
                return labelA.localeCompare(labelB, undefined, { sensitivity: 'base', numeric: true });
            });

            if (this.sortMode === 'alpha_desc') {
                sorted.reverse();
            }

            return sorted;
        }
    },

    methods: {
        isParamHidden(param) {
            if (!param || param.hidden === undefined || param.hidden === null) {
                return false;
            }

            if (typeof param.hidden === 'boolean') {
                return param.hidden;
            }

            if (typeof param.hidden === 'number') {
                return param.hidden !== 0;
            }

            const hiddenVal = String(param.hidden).trim().toLowerCase();
            if (hiddenVal === '') {
                return false;
            }

            if (['visible', 'false', '0', 'no', 'off'].includes(hiddenVal)) {
                return false;
            }

            return ['hidden', 'true', '1', 'yes', 'on'].includes(hiddenVal);
        },

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
parametersRoot.parametersApp = parametersApp;

/**
 * Bridge function to update parameters from jQuery code
 * Called when task parameters are loaded/updated
 */
parametersRoot.updateVueParameters = function(parametersDesc) {
    parametersApp.instance.updateParameters(parametersDesc);
};

/**
 * Bridge function to get parameter values from Vue
 * Called when submitting form
 */
parametersRoot.getVueParameterValues = function() {
    return parametersApp.instance.getParameterValues();
};

/**
 * Bridge function to enable/disable parameter editing
 */
parametersRoot.setVueParametersEditMode = function(enabled) {
    parametersApp.instance.setEditMode(enabled);
};

function Parameters(editable=false) {
    this.obj = document.createElement("table");
    this.traits = {};
    this.editable = editable;
    this.syncVue = false;
}

Parameters.prototype.update = function(desc) {
    if (!desc || typeof desc !== 'object') {
        console.warn("Parameters.update called with invalid desc:", desc);
        if (this.syncVue && typeof parametersRoot !== 'undefined' && typeof parametersRoot.updateVueParameters === 'function') {
            parametersRoot.updateVueParameters({});
        }
        return;
    }

    for (var name in desc) {
        if (typeof(this.traits[name]) != "undefined" && typeof(desc[name].value) == "undefined") {
            var trait = this.traits[name];
            if (trait.inputs.length > 1) {
                var any = false;
                var tuple = [];
                for (var i = 0; i < trait.inputs.length; i++) {
                    tuple.push(trait.inputs[i].value);
                    if (trait.inputs[i].value) {
                        any = true;
                    }
                }
                if (any)
                    desc[name].value = tuple;
            } else if (desc[name]['type'] == 'Bool') {
                desc[name].value = trait.inputs[0].checked;
            } else {
                desc[name].value = trait.inputs[0].value;
            }
        }
    }

    this.obj.innerHTML = "";
    this.traits = {};
    this.hidden_parameters = [];
    this.append(desc);
    this.show_all_attrs();

    if (this.syncVue && typeof parametersRoot !== 'undefined' && typeof parametersRoot.updateVueParameters === 'function') {
        parametersRoot.updateVueParameters(desc);
    }
};

Parameters.prototype.append = function(desc) {
    var funcs = {
        "Float": this.add_float,
        "Int": this.add_int,
        "Tuple": this.add_tuple,
        "Array": this.add_array,
        "Instance": this.add_instance,
        "InstanceFromDB": this.add_instance,
        "DataFile": this.add_instance,
        "String": this.add_string,
        "Enum": this.add_enum,
        "OptionsList": this.add_enum,
        "Bool": this.add_bool,
        "List": this.add_list,
    };

    for (var name in desc) {
        if (funcs[desc[name]['type']]) {
            var fn = funcs[desc[name]['type']].bind(this);
            fn(name, desc[name]);
        } else if (typeof debug === 'function') {
            debug(desc[name]['type']);
        }
    }
};

Parameters.prototype.show_all_attrs = function() {
    var showAll = this.syncVue ? !!parametersApp.instance.showAllParams : false;

    for (var name in this.hidden_parameters) {
        if (showAll)
            $(this.hidden_parameters[name]).show();
        else
            $(this.hidden_parameters[name]).hide();
    }
};

Parameters.prototype.enable = function() {
    if (this.syncVue && typeof parametersRoot !== 'undefined' && typeof parametersRoot.setVueParametersEditMode === 'function') {
        parametersRoot.setVueParametersEditMode(true);
    }
    $(this.obj).find("input, select, checkbox").removeAttr("disabled");
};

Parameters.prototype.disable = function() {
    if (this.syncVue && typeof parametersRoot !== 'undefined' && typeof parametersRoot.setVueParametersEditMode === 'function') {
        parametersRoot.setVueParametersEditMode(false);
    }
    $(this.obj).find("input, select, checkbox").attr("disabled", "disabled");
};

Parameters.prototype.add_to_table = function(name, info) {
    let desc = info["desc"];
    let hidden = info["hidden"];
    let label_text = info["label"];

    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    td.style.textAlign = "right";

    if (label_text != undefined) {
        label.innerHTML = label_text;
    } else {
        label.innerHTML = name;
    }

    label.setAttribute("for", "param_"+name);
    td.appendChild(label);

    if (this.editable && !info["required"]) {
        var remove_row = document.createElement("input");
        remove_row.setAttribute("class", "paramremove");
        remove_row.setAttribute("type", "button");
        remove_row.setAttribute("value", "-");
        var this_ = this;
        $(remove_row).on("click", function() {this_.remove_row(name);});
        td.appendChild(remove_row);
    }

    if (hidden === 'hidden') {
        this.hidden_parameters[name] = $(label).closest("tr");
    }

    return trait;
};

Parameters.prototype.add_tuple = function(name, info) {
    var len = info['default'].length;
    var trait = this.add_to_table(name, info);
    var wrapper = document.createElement("td");
    trait.appendChild(wrapper);
    this.obj.appendChild(trait);

    this.traits[name] = {"obj":trait, "inputs":[]};

    for (var i=0; i < len; i++) {
        var input = document.createElement("input");
        input.type = "text";
        input.name = name;
        input.placeholder = JSON.stringify(info['default'][i]);
        if (typeof(info['value']) != "undefined")
            if (typeof(info['value'][i]) != "string")
                input.value = JSON.stringify(info['value'][i]);
            else
                input.value = info['value'][i];
        if (input.value == input.placeholder)
            input.value = null;
        wrapper.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name].inputs[0].id = "param_"+name;
    for (var j in this.traits[name].inputs) {
        var inputs = this.traits[name].inputs;
        this.traits[name].inputs[j].onchange = function() {
            if (this.value.length > 0) {
                for (var k in inputs)
                    if (inputs[k].placeholder.length == 0) inputs[k].required = "required";
            } else {
                for (var k2 in inputs)
                    inputs[k2].removeAttribute("required");
            }
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
        this.traits[name].inputs[j].onchange();
    }
};

Parameters.prototype.add_int = function(name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    if (typeof(info['value']) != "undefined")
        input.value = info['value'];
    else
        input.value = info['default'];
    if (info['required']) {
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
        input.onchange();
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_float = function(name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.pattern = "-?[0-9]*\\.?[0-9]*";
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info.value;
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info.value);
    if (input.value == input.placeholder)
        input.value = null;
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
        input.onchange();
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_bool = function(name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "checkbox";
    input.name = name;
    input.id = "param_"+name;
    if (typeof(info['value']) != "undefined")
        input.checked=info['value'];
    else
        input.checked = info['default'];
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_array = function(name, info) {
    if (info['default'].length < 4) {
        this.add_tuple(name, info);
        for (var i=0; i < this.traits[name].inputs.length; i++)
            this.traits[name].inputs[i].pattern = '[0-9\\(\\)\\[\\]\\.\\,\\s\\-]*';
    } else {
        this.add_list(name, info);
    }
};

Parameters.prototype.add_string = function(name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    $(input).addClass("string");
    input.name = name;
    input.id = "param_"+name;
    input.placeholder = info['default'];
    if (typeof(info['value']) != "undefined") {
         input.setAttribute("value", info['value']);
    }
    if (input.value == input.placeholder)
        input.value = null;
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
        input.onchange();
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_instance = function(name, info) {
    var options = info['options'];
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("select");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    if (info['required']) {
        var opt = document.createElement("option");
        opt.setAttribute("selected", "selected");
        input.appendChild(opt);
        input.required = "required";
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
    }
    for (var i = 0; i < options.length; i++) {
        var opt2 = document.createElement("option");
        opt2.value = options[i][0];
        opt2.innerHTML = options[i][1];
        if (!info['required'] &&
        (typeof(info['value']) != "undefined" && info['value'] == opt2.value))
            opt2.setAttribute("selected", "selected");
        else if (!info['required'] && typeof(info['value']) == "undefined" && info['default'] == opt2.value)
            opt2.setAttribute("selected", "selected");
        else if (info['required'] && info['default'] == opt2.value)
            opt2.setAttribute("selected", "selected");
        input.appendChild(opt2);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_enum = function(name, info) {
    var options = info['options'];
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("select");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    if (info['required']) {
        var opt = document.createElement("option");
        opt.setAttribute("selected", "selected");
        input.appendChild(opt);
        input.required = "required";
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
    }
    for (var i = 0; i < options.length; i++) {
        var opt2 = document.createElement("option");
        opt2.value = options[i];
        opt2.innerHTML = options[i];
        if (!info['required'] &&
            (typeof(info['value']) != "undefined" && info['value'] == opt2.value))
            opt2.setAttribute("selected", "selected");
        else if (!info['required'] && typeof(info['value']) == "undefined" && info['default'] == opt2.value)
            opt2.setAttribute("selected", "selected");
        else if (info['required'] && info['default'] == opt2.value)
            opt2.setAttribute("selected", "selected");
        input.appendChild(opt2);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
};

Parameters.prototype.add_list = function(name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.is_list = true;
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info['value'];
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info['value']);
    if (input.value == input.placeholder)
        input.value = null;
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        };
        input.onchange();
    }
    input.pattern = /[0-9\(\)\[\]\.\,\s\-]*/;
    this.traits[name] = {"obj":trait, "inputs":[input], "list":true};
};

Parameters.prototype.add_row = function() {
    var trait = $("<tr>");
    trait.attr("title", "New metadata entry");
    var td = $("<td>");
    td.addClass("param_label");
    td.css("textAlign", "right");
    trait.append(td);
    var label = $('<input type="text" placeholder="New entry">');
    label.prop("required",true);
    label.css({"border": "none", "border-color": "transparent"});
    var div = $("<td>");
    var input = $('<input type="text" class="string" required>');
    input.on("change", function() {
        if (this.value.length == 0)
            this.required = "required";
        else if (this.required)
            this.removeAttribute("required");
    });
    td.append(label);
    trait.append(div);
    div.append(input);
    $(this.obj).append(trait);

    var _this = this;
    label.blur(function(){
        var name = label.val();
        if (name) {
            trait.attr('id', 'param_'+name);
            var new_label = $("<label>");
            new_label.css("textAlign", "right");
            new_label.addClass("string");
            new_label.html(name);
            label.replaceWith(new_label);
            _this.traits[name] = {"obj":trait.get(0), "inputs":[input.get(0)]};
            if (_this.editable) {
                var remove_row = document.createElement("input");
                remove_row.setAttribute("class", "paramremove");
                remove_row.setAttribute("type", "button");
                remove_row.setAttribute("value", "-");
                $(remove_row).on("click", function() {_this.remove_row(name);});
                new_label.after(remove_row);
            }
        } else {
            trait.remove();
        }
    });
};

Parameters.prototype.remove_row = function(name) {
    if (typeof(this.traits[name]) != "undefined") {
        var trait = this.traits[name];
        trait.obj.remove();
        delete this.traits[name];
    }
};

function get_param_input(input_obj) {
    if (input_obj.type == 'checkbox') {
        return input_obj.checked;
    } else if (input_obj.is_list && input_obj.value.length > 0) {
        var list = input_obj.value.replace(/\[|\]/g,"").split(/[ ,]+/);
        if (Array.isArray(list)) return list;
        else return [list];
    } else if (input_obj.is_list) {
        var list2 = input_obj.placeholder;
        if (Array.isArray(list2)) return list2;
        else return [list2];
    } else if (input_obj.value.length > 0) {
        return input_obj.value;
    } else {
        return input_obj.placeholder;
    }
}

Parameters.prototype.to_json = function() {
    if (this.syncVue) {
        var vueValues = parametersRoot.getVueParameterValues();
        if (vueValues && typeof vueValues === 'object' && Object.keys(vueValues).length > 0) {
            return vueValues;
        }
    }

    var jsdata = {};

    for (var name in this.traits) {
        var trait = this.traits[name];
        if (trait.inputs.length > 1) {
            var plist = [];
            for (var i = 0; i < trait.inputs.length; i++) {
                plist.push(get_param_input(trait.inputs[i]));
            }
            jsdata[name] = plist;
        } else {
            jsdata[name] = get_param_input(trait.inputs[0]);
        }
    }
    return jsdata;
};

Parameters.prototype.clear_all = function() {
    for (var name in this.traits) {
        var trait = this.traits[name];
        for (var i = 0; i < trait.inputs.length; i++) {
            trait.inputs[i].value = null;
            if (trait.inputs[i].onchange)
                trait.inputs[i].onchange();
        }
    }
};

parametersRoot.Parameters = Parameters;

if (typeof(module) !== 'undefined' && module.exports) {
  exports.Parameters = Parameters;
  exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}
