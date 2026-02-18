/**
 * TaskEntry run lifecycle compatibility module
 * Preserves TaskEntry run/start/stop behavior after modular split from list-vue.js
 */

function stop_fn_callback(resp) {
    debug("Stop callback received");
}

TaskEntry.prototype.stop = function() {
    debug("TaskEntry.prototype.stop");
    var csrf = {};
    csrf['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
    $.post("stop/", csrf, task_interface.trigger.bind(this));
};

TaskEntry.prototype.test = function() {
    debug("TaskEntry.prototype.test");
    return this.run(false, true);
};

TaskEntry.prototype.start = function() {
    debug("TaskEntry.prototype.start");
    return this.run(true, true);
};

TaskEntry.prototype.saverec = function() {
    return this.run(true, false);
};

TaskEntry.prototype.run = function(save, exec) {
    debug("TaskEntry.run");
    task_interface.trigger.bind(this)({status: "stopped"});

    let valid = true;
    $('[required]').each(function() {
        if ($(this).is(':invalid') || !$(this).val()) valid = false;
    });
    if (!valid) {
        $("#experiment").trigger("submit");
        return;
    }

    if (this.report){
        this.report.destroy();
    }
    this.report = new Report(task_interface.trigger.bind(this));
    this.files.hide();
    this.disable();

    var form = {};
    form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
    form['data'] = JSON.stringify(this.get_data());

    var post_url = "";
    if (save && exec) {
        post_url = "/start";
    } else if (save && !exec) {
        post_url = "/saverec";
    } else if (!save && exec) {
        post_url = "/test";
    }

    $.post(post_url, form,
        function(info) {
            task_interface.trigger.bind(this)(info);
            this.report.update(info);
            if (info.status == "running") {
                this.new_row(info);
                this.start_button_pressed = true;
                this.report.activate();
                if (reportVueApp.instance) reportVueApp.instance.activate();
                this.report.set_mode("running");
            } else if (info.status == "completed") {
                this.new_row(info);
                this.tr.removeClass("running active error testing");
                this.destroy();
                te = new TaskEntry(this.idx);
                te.tr.addClass("active");
            }
       }.bind(this)
    );
    return false;
};
