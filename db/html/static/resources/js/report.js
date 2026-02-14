// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(`    <fieldset id="report">
          <legend>Report</legend>
          <div id="report_div">
            <input type="button" value="Update report" id="report_update" onclick="$.post('report', {}, function(info) {te.report.update(info['data']); debug(info);})"><br>
            <input type="button" value="Save FPS Log" id="save_fps_log" onclick="te.report.save_fps_log()"><br>
            <table class="option" id="report_info">
            </table>

            <div class="report_table" id="report_msgs">
              <pre id="stdout"></pre>
            </div>


            <div class="clear"></div>
          </div>
        </fieldset>    `);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}


function Report(callback) {
    // store a ref to the callback function passed in
    this.notify = callback;

    // this.info is a summary stat table
    this.info = $("#report_info");

    // this.msgs = text printed by the task
    this.msgs = $("#report_msgs");

    // Used for error messages?
    this.stdout = $("#stdout");

    this.boxes = {};
    this.ws = null;
}

Report.prototype.activate = function() {
    if (!this.ws) {
        // Create a new JS WebSocket object
        // Connect to Django Channels WebSocket endpoint
        var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        var host = window.location.host;
        this.ws = new WebSocket(protocol + "//" + host + "/ws/tasks/");

        this.ws.onopen = function() {
            console.log("Report WebSocket connected");
        };

        this.ws.onmessage = function(evt) {
            try {
                debug(evt.data);
                var data = JSON.parse(evt.data);
                
                // Handle different message types from Django Channels
                if (data.type === 'task_status') {
                    // Update report with the reportstats data
                    this.update(data.reportstats || {});
                } else {
                    // Fallback for other message formats
                    this.update(data);
                }
            } catch(e) {
                console.error("Error processing WebSocket message:", e);
            }
        }.bind(this);

        this.ws.onerror = function(error) {
            console.error("Report WebSocket error:", error);
        };

        this.ws.onclose = function() {
            console.log("Report WebSocket disconnected");
            this.ws = null;
            // Try to reconnect after 2 seconds
            setTimeout(function() {
                if (this && this.ws === null) {
                    console.log("Attempting to reconnect WebSocket...");
                    this.activate();
                }
            }.bind(this), 2000);
        }.bind(this);
    }
}
Report.prototype.fps_log = [];

Report.prototype.update = function(info) {
    // Guard against undefined/null info
    if (!info || typeof info !== 'object') {
        console.warn("Report.update called with invalid info:", info);
        return;
    }
    
    // run the 'notify' callback every time this function is provided with info
    if (typeof(this.notify) == "function" && !$.isEmptyObject(info))
        this.notify(info);

    if (info.status && info.status == "error") { // received an error message through the websocket
        // append the error message (pre-formatted by python traceback) onto the printed out messages
        this.msgs.append("<pre>"+info.msg+"</pre>");
    } else if (info.status && info.status == "stdout") {
        this.stdout.append(info.msg);
    } else {
        for (var stat in info) {
            if (!this.boxes[stat]) { // if we haven't already made a table row for this stat
                if (!stat.match("rig_name|status|task|subj|date|idx")) { // if this is not one of the stats we ignore because it's reported elsewhere
                    var row = document.createElement("tr");

                    // make a column in the row for the stat name
                    var label = document.createElement("td");
                    label.innerHTML = stat;

                    // make a column in the row to hold the data to be updated by the server
                    var data = document.createElement("td");

                    row.appendChild(label);
                    row.appendChild(data);

                    this.info.append(row);

                    // save ref to the 'data' box, to be updated when new 'info' comes in
                    this.boxes[stat] = data;
                }
            }
        }

        // Update the stat data
        for (var stat in this.boxes) {
            if (info[stat])
                this.boxes[stat].innerHTML = info[stat];
        }

        for (var stat in this.boxes) {
            if (info[stat]) {
                this.boxes[stat].innerHTML = info[stat];
                // Save FPS to log if it's present
                if (stat.toLowerCase().includes("fps")) {
                    this.fps_log.push(info[stat]);
                }
            }
        }
    }
}

Report.prototype.save_fps_log = function() {
    console.log("FPS log:", this.fps_log);

    // download as a file
    const blob = new Blob([this.fps_log.join("\n")], {type: "text/plain"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "fps_log.txt";
    a.click();
    URL.revokeObjectURL(url);
}

Report.prototype.manual_update = function() {
    $.post('report', {}, function(info) {te.report.update(info['data']);});
}

Report.prototype.destroy = function () {
    this.deactivate();
    this.msgs.html('<pre id="stdout"></pre>');
    this.info.html("");
}

Report.prototype.deactivate = function() {
    /*
        Close the report websocket
    */
    if (this.ws)
        this.ws.close();
    delete this.ws;
}
Report.prototype.hide = function() {
    $("#report").hide();
}

Report.prototype.show = function() {
    $("#report").show();
}

Report.prototype.set_mode = function(mode) {
    if (mode == "completed") {
        $("#report_update").hide();
    } else if (mode == "running") {
        $("#report_update").show();
    }
}

if (typeof(module) !== 'undefined' && module.exports) {
  exports.Report = Report;
  exports.$ = $;
}
