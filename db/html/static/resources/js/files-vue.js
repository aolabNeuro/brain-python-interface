/**
 * Files compatibility module
 * Preserves the Files API used by list-vue.js
 */

const filesRoot = typeof window !== 'undefined' ? window : globalThis;

function Files() {
    this.neural_data_found = false;
    $("#file_modal_server_resp").html("");
}

Files.prototype.hide = function() {
    $("#files").hide();
};

Files.prototype.show = function() {
    $("#files").show();
};

Files.prototype.clear = function() {
    $("#file_list").html("");
};

Files.prototype.update_filelist = function(datafiles, task_entry_id) {
    var numfiles = 0;
    this.filelist = document.createElement("ul");

    for (var sys in datafiles) {
        for (var i = 0; i < datafiles[sys].length; i++) {
            var file = document.createElement("li");
            file.textContent = datafiles[sys][i];
            this.filelist.appendChild(file);
            numfiles++;
        }
    }

    if (numfiles > 0) {
        $("#file_list").append(this.filelist);
        for (var system in datafiles) {
            if (system == "plexon" || system == "blackrock" || system == "ecube") {
                this.neural_data_found = true;
                break;
            }
        }
    }
};

filesRoot.Files = Files;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.Files = Files;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}