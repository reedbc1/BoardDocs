const fs = require("fs");
const cheerio = require("cheerio");

async function getBoardIds() {
    const response = await fetch("https://go.boarddocs.com/mo/slcl/Board.nsf/BD-GETMeetingsListForSEO?open&0.6159169630587711", {
    "headers": {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": "SessionID=A475F12D893932AE307737759F211C77DEC93127",
        "Referer": "https://go.boarddocs.com/mo/slcl/Board.nsf/Public"
    },
    "body": null,
    "method": "GET"
    });

    const data = await response.json();
    const jsonData = JSON.stringify(data);
    await writeFile('json_files/meetingIds.json', jsonData);
    return data;
}


async function writeFile(fileName, jsonData) {
    return new Promise((resolve, reject) => {
        fs.writeFile(fileName, jsonData, function(err) {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
}


function readFile(filename) {
    const fs = require('node:fs');
    try {
        const data = fs.readFileSync(filename, 'utf8');
        return data;
    } catch (err) {
        console.error(`${filename} not found.`);
    }

}


async function downloadMinutes(id) {
    const response = await fetch("https://go.boarddocs.com/mo/slcl/Board.nsf/BD-GetMinutes?open&login0.4331538422449709", {
    "headers": {
        "accept": "text/html, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": "SessionID=A475F12D893932AE307737759F211C77DEC93127",
        "Referer": "https://go.boarddocs.com/mo/slcl/Board.nsf/Public"
    },
    "body": `id=${id}&current_committee_id=AAUJEW4CE322`,
    "method": "POST"
    });

    try {
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }
        const html = await response.text();
        return html;
    } catch (error) {
        console.error(`Failed to download minutes for ${id}: ${error.message}`);
        return null;
    }
}


async function main() {
    // Load processed IDs
    let processedIds = [];
    try {
        const data = readFile('json_files/processedIds.json');
        processedIds = JSON.parse(data);
    } catch (e) {
        // File doesn't exist, start with empty array
    }

    // Fetch meeting IDs
    const meetingData = await getBoardIds();

    // Extract all IDs
    const allIds = meetingData.map(arr => {
        return arr.Unique;
    });

    // Filter to only new IDs
    const newIds = allIds.filter(id => !processedIds.includes(id));

    // Process new IDs
    for (let id of newIds) {
        console.log(`Processing id: ${id}`);

        // Download minutes
        const minutesHtml = await downloadMinutes(id.unique);
        if (minutesHtml) {
            const text = extractText(minutesHtml);
            if (text.trim()) {
                const sanitizedDate = id.date.split('T')[0]; // Gets '2025-12-18'
                const metadata = JSON.stringify({
                    name: id.name,
                    description: id.description,
                    unique: id.unique,
                    date: id.date
                }, null, 2);
                const contentWithMetadata = `${metadata}\n\n---\n\n${text}`;
                const textPath = `minutes\\${sanitizedDate}.txt`;
                await writeFile(textPath, contentWithMetadata);
                console.log(`Saved minutes text for ${id.date}`);
            } else {
                console.log(`No text in minutes for ${id.date}`);
            }
        }
    }

    // Update processed IDs
    processedIds.push(...newIds);
    await writeFile('json_files/processedIds.json', JSON.stringify(processedIds));
}

main().catch(console.error);
console.log(downloadMinutes("9GX2PP6A2F54"))

function extractText(html) {
    
    // load into cheerio
    const $ = cheerio.load(html);

    // get text only
    let text = $("body").text();

    // clean it up
    text = text
    .replace(/\u00a0/g, " ")        // non-breaking spaces
    .replace(/\s+\n/g, "\n")        // trailing spaces
    .replace(/\n{3,}/g, "\n\n")     // collapse blank lines
    .trim();

    return text;
}

