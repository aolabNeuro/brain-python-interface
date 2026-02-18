/**
 * TaskEntry utility compatibility module
 * Preserves TaskEntry utility methods split from list-vue.js
 */

TaskEntry.prototype.new_row = function(info) {
    debug('TaskEntry.prototype.new_row: ' + info.idx);

    if (typeof(info.idx) == "number") {
        this.idx = info.idx;
    } else {
        this.idx = parseInt(info.idx.match(/\((\d+)\)/)[1]);
    }
    this.tr.removeClass("running active error testing");

    this.tr.hide();
    this.tr.click(function() {
        if (te) te.destroy();
        te = new TaskEntry(null);
    });

    feats.unbind_change_callback();
    $("#tasks").unbind("change");

    this.tr = $(document.createElement("tr"));
    this.tr.attr("id", "row" + this.idx);

    this.tr.html("<td class='colDate'>Now</td>" +
                "<td class='colTime' >--</td>" +
                "<td class='colID'   >" + info.idx + "</td>" +
                "<td class='colRig'  >" + info.rig_name + "</td>" +
                "<td class='colSubj' >" + info.subj + "</td>" +
                "<td class='colTask' >" + info.task + "</td>");

    $("#newentry").after(this.tr);
    this.tr.addClass("active rowactive running");
    this.tr.find('td').addClass("firstRowOfday");
    this.tr.next().find('td').removeClass("firstRowOfday");
    this.notes = new Notes(this.idx);
};

TaskEntry.prototype.get_data = function() {
    var data = {};
    data['task'] = parseInt($("#tasks").val());
    data['feats'] = feats.get_checked_features();
    data['params'] = this.params.to_json();
    data['metadata'] = this.metadata.get_data();
    data['sequence'] = this.sequence.get_data();
    data['entry_name'] = $("#entry_name").val();
    data['date'] = $("#newentry_today").html();

    return data;
};

TaskEntry.prototype.enable = function() {
    debug("TaskEntry.prototype.enable");
    this.params.enable();
    this.metadata.enable();
    feats.enable_entry();
    if (this.sequence)
        this.sequence.enable();
    if (!this.idx)
        $("#subjects, #tasks").removeAttr("disabled");
};

TaskEntry.prototype.disable = function() {
    debug("TaskEntry.prototype.disable");
    this.params.disable();
    this.metadata.disable();
    feats.disable_entry();
    if (this.sequence)
        this.sequence.disable();
    if (!this.idx)
        $("#subjects, #tasks").attr("disabled", "disabled");
};

TaskEntry.prototype.link_new_files = function() {
    data = {
        "data_system_id": $("#data_system_id").val(),
    };

    var file_path = $("#file_path").val();
    var new_file_path = $("#new_file_path").val();
    var new_file_data = $("#new_file_raw_data").val();
    var new_file_data_format = $("#new_file_data_format").val();
    var browser_sel_file = document.getElementById("file_path_browser_sel").files[0];

    if ($.trim(new_file_data) != "" && $.trim(new_file_path) != "") {
        data['file_path'] = new_file_path;
        data['raw_data'] = new_file_data;
        data['raw_data_format'] = new_file_data_format;
    } else if (file_path != "") {
        data['file_path'] = file_path;
        data['raw_data'] = '';
        data['raw_data_format'] = null;
    } else if (browser_sel_file != undefined) {
        data['file_path'] = browser_sel_file.name;
        data['raw_data'] = '';
        data['raw_data_format'] = null;
    } else {
        data = {};
    }

    $.post("/exp_log/link_data_files/" + this.idx + "/submit", data,
        function(resp) {
            $("#file_modal_server_resp").append(resp + "<br>");
            debug("posted the file!");
        }
    );
};
