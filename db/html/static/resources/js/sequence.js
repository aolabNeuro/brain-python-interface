/**
 * Sequence compatibility module
 * Replaces legacy sequence.js while preserving the Sequence API used by list.js
 */

const sequenceRoot = typeof window !== 'undefined' ? window : globalThis;

if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(`<fieldset id="sequence">
        <div id="seqparams" hidden="true">
            Parameters
        </div>

        <select id="seqlist" name="seq_name">
            <option value="new">Create New...</option>
        </select>

        <select id="seqgen" name="seq_gen" hidden="true">
            <option value="gen1">gen1</option>
            <option value="gen2">gen2</option>
        </select>

        <input id="seqstatic" type="checkbox" name="seqstatic">
        <label for="seqstatic">Static</label>
    </fieldset>`);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}

function Sequence() {
    var params = new Parameters();
    this.params = params;

    var form = $("<form action='javascript:te.sequence.add_sequence()'></form>");
    $(form).append(this.params.obj);
    $("#seqparams").html(form);

    var self = this;
    this._handle_set_new_name = function() {
        $("#seqlist option[value=\"new\"]").attr("name", self._make_name());
    };

    this._handle_chgen = function() {
        $.getJSON("/ajax/gen_info/" + this.value + "/", {}, function(info) {
            params.update(info.params);
            self._handle_set_new_name();
            $("#seqparams input").change(self._handle_set_new_name);

            if ($("#seqlist").val() != "new") {
                $("#seqstatic,#seqparams,#seqparams input, #seqgen").attr("disabled", "disabled");
                $('#seqadd').hide();
            }
        });
    };

    $("#seqgen").change(this._handle_chgen);
    this.options = {};
    this.gens = null;
}

Sequence.prototype.seqlist_type = function() {
    return $("#seqlist").is("select") ? "select" : "input";
};

Sequence.prototype.update = function(info) {
    if (info === undefined) {
        return;
    }

    $("#seqlist").unbind("change");
    var prev = $("#seqlist :selected").val();

    for (var id in this.options) {
        $(this.options[id]).remove();
    }

    $("#seqlist").replaceWith(
        "<select id='seqlist' name='seq_name'><option value='new'>Create New...</option></select>"
    );

    this.options = {};
    var opt;
    var lastId = null;
    for (id in info) {
        opt = document.createElement("option");
        opt.innerHTML = info[id].name;
        opt.value = id;
        this.options[id] = opt;
        $("#seqlist").append(opt);
        lastId = id;
    }

    if (Object.keys(info).length > 0) {
        var activeId;
        if (!(prev in info)) {
            activeId = lastId;
            $("#seqgen").val(info[activeId].generator[0]);
            $("#seqlist").val(activeId);
        } else {
            activeId = prev;
            $("#seqgen").val(info[activeId].generator[0]);
            $("#seqlist").val(activeId);
        }

        this.params.update(info[activeId].params);
        $("#seqstatic").prop("checked", !!info[activeId].static);

        var seqObj = this;
        this._handle_chlist = function() {
            var selectedId = this.value;
            if (selectedId == "new") {
                seqObj.edit();
            } else {
                seqObj.params.update(info[selectedId].params);
                $('#seqgen').val(info[selectedId].generator[0]);
                $("#seqstatic").prop("checked", !!info[selectedId].static);
                $("#seqstatic,#seqparams,#seqparams input, #seqgen").attr("disabled", "disabled");
                $('#seqadd').hide();
            }
        };

        $("#seqlist").change(this._handle_chlist);
        $("#seqlist").change();
        $("#seqstatic,#seqparams,#seqparams input, #seqgen").attr("disabled", "disabled");
        $("#seqadd").hide();
    } else {
        this.edit();
    }
};

Sequence.prototype.destroy_parameters = function() {
    if (this.params) {
        $(this.params.obj).remove();
        delete this.params;
    }
};

Sequence.prototype.destroy = function() {
    for (var id in this.options) {
        $(this.options[id]).remove();
    }

    this.destroy_parameters();
    $("#seqlist").unbind("change");
    $("#seqgen").unbind("change");
};

Sequence.prototype._make_name = function() {
    var gen = $("#sequence #seqgen option").filter(":selected").text();
    var txt = [];
    var data = this.params.to_json();

    for (let key in data) {
        if (Array.isArray(data[key])) {
            txt.push(key + "=" + data[key].join(' '));
        } else {
            txt.push(key + "=" + data[key]);
        }
    }

    var isStatic = $("#seqstatic").prop("checked") ? "static" : "";
    return gen + ":[" + txt.join(", ") + "]" + isStatic;
};

Sequence.prototype.edit = function() {
    $("#seqlist").val("new");
    $("#seqparams input").val("");
    this._handle_set_new_name();
    $("#seqgen, #seqparams, #seqparams input, #seqstatic").removeAttr("disabled");
    $("#seqgen").change();
    $('#seqadd').show();
};

Sequence.prototype.add_sequence = function() {
    var form = {};
    form['task'] = parseInt($("#tasks").attr("value"));
    form['sequence'] = JSON.stringify(this.get_data());

    var self = this;
    $.post('/exp_log/ajax/add_sequence', form, function(resp) {
        if (resp.id) {
            if ($("#seqlist option[value='" + resp.id + "']").length > 0) {
                $("#seqlist").val(resp.id);
                log("Switched to existing sequence", 2);
            } else {
                var opt = document.createElement("option");
                opt.innerHTML = resp.name;
                opt.value = resp.id;
                self.options[resp.id] = opt;
                $("#seqlist").append(opt);
                $("#seqlist").val(resp.id);
                log("Added new sequence", 2);
            }
            te._task_query(function(){}, false, false);
        } else {
            log("Problem adding sequence", 5);
        }
    });
};

Sequence.prototype.enable = function() {
    $("#seqlist").removeAttr("disabled");
    if ($("#seqlist").val() == "new") {
        $("#seqgen, #seqparams, #seqparams input, #seqstatic").removeAttr("disabled");
        $('#seqadd').show();
    }
};

Sequence.prototype.disable = function() {
    $("#seqlist, #seqparams, #seqparams input, #seqgen, #seqstatic").attr("disabled", "disabled");
    $("#seqadd").hide();
};

Sequence.prototype.get_data = function() {
    var val = $("#seqlist").val();
    var name = $("#seqlist option[value=\"new\"]").attr("name");
    var id = null;

    $("#seqlist option").each(function() {
        if ($(this).html() == name) {
            id = parseInt($(this).val());
        }
    });

    if (val == "new" && id != null) {
        return id;
    } else if (val == "new") {
        var data = {};
        data['name'] = name;
        data['generator'] = $("#seqgen").val();
        data['params'] = this.params.to_json();
        data['static'] = $("#seqstatic").prop("checked");
        return data;
    } else {
        return parseInt(val);
    }
};

Sequence.prototype.update_available_generators = function(gens) {
    if (this.gens && JSON.stringify(gens) == JSON.stringify(this.gens)) {
        return;
    }

    this.gens = gens;
    if (gens.length > 0) {
        $('#seqgen').empty();
        for (var i = 0; i < gens.length; i++) {
            $('#seqgen')
                .append($('<option>', { value: gens[i][0] })
                .text(gens[i][1]));
        }
    }
    $('#seqlist').change();
};

sequenceRoot.Sequence = Sequence;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.Sequence = Sequence;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}
