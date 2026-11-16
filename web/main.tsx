import React from "react";
import { createRoot } from "react-dom/client";
import MeetingApp from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MeetingApp />
  </React.StrictMode>,
);
