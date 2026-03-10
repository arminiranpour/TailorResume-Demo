const stringifyValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => stringifyValue(item))
      .filter(Boolean)
      .join(" ")
      .trim();
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return "";
    }
  }
  return "";
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
};

export const normalizeError = (error: unknown): string => {
  if (!error) {
    return "Unexpected error";
  }

  if (typeof error === "string") {
    return error;
  }

  if (isRecord(error)) {
    const payload = isRecord(error.payload) ? error.payload : null;
    if (payload) {
      const payloadMessage = [
        stringifyValue(payload.detail),
        stringifyValue(payload.error),
        stringifyValue(payload.details),
        stringifyValue(payload.reasons),
      ]
        .filter(Boolean)
        .join(" ")
        .trim();
      if (payloadMessage) {
        return payloadMessage;
      }
    }

    const directMessage = stringifyValue(error.message);
    if (directMessage) {
      return directMessage;
    }

    const response = isRecord(error.response) ? error.response : null;
    if (response) {
      const responseData = response.data;
      const responsePayload = isRecord(responseData) ? responseData : null;
      const responseMessage = [
        stringifyValue(responsePayload?.detail ?? responsePayload?.error),
        stringifyValue(responseData),
        stringifyValue(response.statusText),
      ]
        .filter(Boolean)
        .join(" ")
        .trim();
      if (responseMessage) {
        return responseMessage;
      }
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  const fallback = stringifyValue(error);
  return fallback || "Unexpected error";
};

export const toErrorMessage = (value: unknown): string => {
  const message = stringifyValue(value);
  return message || "Unexpected error";
};
