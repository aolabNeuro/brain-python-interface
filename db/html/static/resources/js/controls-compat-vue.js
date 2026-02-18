/**
 * Controls compatibility module
 * Preserves the legacy Controls API used by list-vue.js
 */

const controlsCompatRoot = typeof window !== 'undefined' ? window : globalThis;

function create_control_callback(i, control_str, args, static=false) {
    return function() {trigger_control(i, control_str, args, static)};
}

function trigger_control(i, control, params, static) {
    debug("Triggering control: " + control);
    if (static) {
        var data = {
            "control": control,
            "params": JSON.stringify(params.to_json()),
            "base_class": $('#tasks').val(),
            "feats": JSON.stringify(feats.get_checked_features())
        };
        $.post("trigger_control", data, function(resp) {
            debug("Control response", resp);
            if (resp["status"] == "success") {
                $('#controls_btn_' + i.toString()).css({"background-color": "green"});
                $('#controls_btn_' + i.toString()).animate({"background-color": "black"}, 500);
            }
        });
    } else {
        $.post("trigger_control", {"control": control, "params": JSON.stringify(params.to_json())}, function(resp) {
            debug("Control response", resp);
            params.clear_all();
            if (resp["status"] == "pending") {
                $('#controls_btn_' + i.toString()).css({"background-color": "yellow"});
                $('#controls_btn_' + i.toString()).animate({"background-color": "black"}, 500);
            }
        });
    }
}

function Controls() {
    this.control_list = [];
    this.static_control_list = [];
    this.params_list = [];
    this.static_params_list = [];
}

Controls.prototype.update = function(controls) {
    debug("Updating controls");

    if (!controls || !Array.isArray(controls)) {
        console.warn("Controls.update called with invalid controls:", controls);
        return;
    }

    $("#controls_table").html('');
    this.control_list = [];
    this.static_control_list = [];
    this.params_list = [];
    this.static_params_list = [];
    for (var i = 0; i < controls.length; i += 1) {
        var new_params = new Parameters();
        new_params.update(controls[i].params);

        var new_button = $('<button/>',
            {
                text: controls[i].name,
                id: "controls_btn_" + i.toString(),
                click: create_control_callback(i, controls[i].name, new_params, controls[i].static),
                type: "button"
            }
        );

        $("#controls_table").append(new_button);
        $("#controls_table").append(new_params.obj);

        if (controls[i].static) {
            this.static_control_list.push(new_button);
            this.static_params_list.push(new_params);
        } else {
            this.control_list.push(new_button);
            this.params_list.push(new_params);
        }
    }

    if (this.control_list.length > 0) {
        this.show();
    } else {
        this.hide();
    }
};

Controls.prototype.hide = function() {
    $("#controls").hide();
};

Controls.prototype.show = function() {
    if (this.control_list.length > 0) {
        $("#controls").show();
    }
    this.deactivate();
};

Controls.prototype.activate = function() {
    for (var i = 0; i < this.control_list.length; i += 1) {
        $(this.control_list[i]).prop('disabled', false);
    }
    for (var j = 0; j < this.params_list.length; j += 1) {
        this.params_list[j].enable();
    }
};

Controls.prototype.deactivate = function() {
    for (var i = 0; i < this.control_list.length; i += 1) {
        $(this.control_list[i]).prop('disabled', true);
    }
    for (var j = 0; j < this.params_list.length; j += 1) {
        this.params_list[j].disable();
    }
};

controlsCompatRoot.Controls = Controls;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.Controls = Controls;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}