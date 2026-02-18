/**
 * Notes compatibility module
 * Keeps the legacy Notes API while list-vue.js is incrementally decomposed.
 */

const notesRoot = typeof window !== 'undefined' ? window : globalThis;

function Notes(idx) {
    this.idx = idx;
    $("#notes").val("");
    if (typeof debug === 'function') {
        debug("Cleared notes");
    }
    this.activate();
}

Notes.prototype.update = function(notes) {
    if (!notes) {
        console.warn("Notes.update called with invalid notes:", notes);
        notes = "";
    }
    $("#notes textarea").attr("value", notes);
};

Notes.prototype.activate = function() {
    var notes_keydown_handler = function() {
        if (this.last_TO != null) {
            clearTimeout(this.last_TO);
        }
        this.last_TO = setTimeout(this.save.bind(this), 500);
    }.bind(this);
    $("#notes textarea").keydown(notes_keydown_handler);
};

Notes.prototype.destroy = function() {
    $("#notes textarea").unbind("keydown");

    if (this.last_TO != null) {
        clearTimeout(this.last_TO);
    }
    this.save();

    $("#notes textarea").val("").removeAttr("disabled");
};

Notes.prototype.save = function() {
    this.last_TO = null;
    var notes_data = {
        "notes": $("#notes textarea").attr("value"),
        'csrfmiddlewaretoken': $("#experiment input[name=csrfmiddlewaretoken]").attr("value")
    };
    $.post("ajax/save_notes/" + this.idx + "/", notes_data);
    if (typeof debug === 'function') {
        debug("Saved notes");
    }
};

notesRoot.Notes = Notes;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.Notes = Notes;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}
