const getStorage = (): Storage | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  const storage = window.localStorage;
  if (
    !storage ||
    typeof storage.getItem !== 'function' ||
    typeof storage.setItem !== 'function' ||
    typeof storage.removeItem !== 'function'
  ) {
    return null;
  }

  return storage;
};

export const safeLocalStorage = {
  getItem(key: string): string | null {
    const storage = getStorage();
    if (!storage) {
      return null;
    }

    try {
      return storage.getItem(key);
    } catch (error) {
      console.warn(`Failed to read localStorage key "${key}"`, error);
      return null;
    }
  },

  setItem(key: string, value: string): boolean {
    const storage = getStorage();
    if (!storage) {
      return false;
    }

    try {
      storage.setItem(key, value);
      return true;
    } catch (error) {
      console.warn(`Failed to write localStorage key "${key}"`, error);
      return false;
    }
  },

  removeItem(key: string): boolean {
    const storage = getStorage();
    if (!storage) {
      return false;
    }

    try {
      storage.removeItem(key);
      return true;
    } catch (error) {
      console.warn(`Failed to remove localStorage key "${key}"`, error);
      return false;
    }
  }
};
