/**
 * TaskInterface compatibility module
 * Preserves the task interface state machine used by list-vue.js
 */

const taskInterfaceCompatRoot = typeof window !== 'undefined' ? window : globalThis;

function interface_fn_completed() {
    log("state = completed", 2);
    $(window).unbind("unload");
    this.tr.removeClass("running error testing").addClass("rowactive active");
    $("#content").removeClass("running error testing");
    this.disable();
    $("#start_buttons").hide();
    $("#stop_buttons").hide();
    $("#finished_task_buttons").show();

    $("#report").show();
    $("#notes").show();
    this.controls.show();
    this.controls.deactivate();
    this.report.deactivate();
    if (reportVueApp.instance) reportVueApp.instance.deactivate();
    this.report.set_mode("completed");

    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }

    if (this.start_button_pressed) setTimeout(function(){ te.reload(); }, 1000);
}

function interface_fn_stopped() {
    log("state = stopped", 2);
    $(window).unbind("unload");
    $("#content").removeClass("running error testing");
    this.tr.removeClass("running error testing").addClass("rowactive active");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();

    $("#report").show();
    $("#notes").hide();
    this.controls.deactivate();

    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }
}

function interface_fn_running(info) {
    log("state = running", 2);
    $(window).unbind("unload");
    this.tr.removeClass("error testing").addClass("running");
    $('#content').removeClass("error testing").addClass("running");
    this.disable();
    $("#stop_buttons").show();
    $("#start_buttons").hide();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.activate();
    if (reportVueApp.instance) reportVueApp.instance.activate();

    $("#report").show();
    $("#notes").show();
    this.controls.show();
    this.controls.activate();

    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }
}

function interface_fn_testing(info) {
    log("state = testing", 2);
    $(window).unload(te.stop);

    this.tr.removeClass("error running").addClass("testing");
    $('#content').removeClass("error running").addClass("testing");
    te.disable();

    $("#stop_buttons").show();
    $("#start_buttons").hide();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.activate();
    if (reportVueApp.instance) reportVueApp.instance.activate();
    if (window.updateVueTaskState) window.updateVueTaskState('testing');

    $("#report").show();
    $("#notes").hide();
    this.controls.show();
    this.controls.activate();
}

function interface_fn_error(info) {
    log("state = error", 2);
    $(window).unbind("unload");
    this.tr.removeClass("running testing").addClass("error");
    $('#content').removeClass("running testing").addClass("error");
    this.disable();
    $("#start_buttons").hide();
    $("#finished_task_buttons").show();
    $("#bmi").hide();
    this.report.deactivate();
    if (reportVueApp.instance) reportVueApp.instance.deactivate();
    if (window.updateVueTaskState) window.updateVueTaskState('error');

    $("#report").show();
    this.controls.deactivate();
}

function interface_fn_errtest(info) {
    log("state = errtest", 2);

    $(window).unbind("unload");
    this.tr.removeClass("running testing").addClass("error");
    $('#content').removeClass("running testing").addClass("error");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.deactivate();
    if (reportVueApp.instance) reportVueApp.instance.deactivate();
    if (window.updateVueTaskState) window.updateVueTaskState('errtest');

    $("#report").show();
    this.controls.deactivate();
}

function TaskInterfaceConstructor() {
    if (typeof debug === 'function') {
        debug("TaskInterfaceConstructor");
    }
    var state = "";
    var lastentry = null;

    this.trigger = function(info) {
        debug("TaskInterfaceConstructor.trigger");
        debug(this);
        debug(info);
        if (this != lastentry) {
            debug(2);
            if (lastentry && !lastentry.destroyed) {
                $(window).unload();
                lastentry.destroy();
            }
            state = this.status;
            states[state].bind(this)(info);
            lastentry = this;
        }

        var transitions = fsm_transition_table[state];
        for (var next_state in transitions) {
            let _test_next_state = transitions[next_state].bind(this);
            if (_test_next_state(info)) {
                debug("executing transition...");
                debug(info);
                let _start_next_state = states[next_state].bind(this);
                _start_next_state(info);
                this.status = next_state;
                state = next_state;
                return;
            }
        }
        debug("No transition found!");
    };

    var fsm_transition_table = {
        "completed": {
            stopped: function(info) { return this.idx == null; },
            running: function(info) { return info.status == "running"; },
            testing: function(info) { return info.status == "testing"; },
            error: function(info) { return info.status == "error"; }
        },
        "stopped": {
            running: function(info) { return info.status == "running"; },
            testing: function(info) {return info.status == "testing"; },
            errtest: function(info) { return info.status == "error"; }
        },
        "running": {
            error: function(info) { return info.status == "error"; },
            completed: function(info) { return info.State == "stopped" || info.status == "stopped"; },
        },
        "testing": {
            errtest: function(info) { return info.status == "error"; },
            stopped: function(info) { return info.State == "stopped" || info.status == "stopped"; },
        },
        "error": {
            running: function(info) { return info.status == "running"; },
            testing: function(info) { return info.status == "testing"; },
            stopped: function(info) { return info.status == "stopped"; },
        },
        "errtest": {
            running: function(info) { return info.status == "running"; },
            testing: function(info) { return info.status == "testing"; },
            stopped: function(info) { return info.status == "stopped"; },
        },
    };

    var states = {
        "completed": interface_fn_completed,
        "stopped": interface_fn_stopped,
        "running": interface_fn_running,
        "testing": interface_fn_testing,
        "error": interface_fn_error,
        "errtest": interface_fn_errtest,
    };
}

var task_interface = new TaskInterfaceConstructor();

taskInterfaceCompatRoot.TaskInterfaceConstructor = TaskInterfaceConstructor;
taskInterfaceCompatRoot.task_interface = task_interface;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.TaskInterfaceConstructor = TaskInterfaceConstructor;
    exports.task_interface = task_interface;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}