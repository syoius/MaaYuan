import fetch from "node-fetch";

export default async function handler(req, res) {
  const targetUrl = "https://docs.qq.com/aio/DS1BMQmpiQkdOb1RT";

  try {
    const response = await fetch(targetUrl);
    const html = await response.text();

    res.setHeader("Content-Type", "text/html");
    res.status(200).send(html);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error fetching target URL");
  }
}
