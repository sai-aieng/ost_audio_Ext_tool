import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { OutputPage } from "./OutputPage";
import "./styles.css";

const Page = window.location.pathname.replace(/\/$/, "") === "/output" ? OutputPage : App;
ReactDOM.createRoot(document.getElementById("root")).render(<React.StrictMode><Page /></React.StrictMode>);