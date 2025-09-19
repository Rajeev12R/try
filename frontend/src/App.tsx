import { Routes, Route } from "react-router-dom";
import Admin from "./pages/Admin";
import Home from "./pages/Home";

const App = () => {
  return (
    <div>
      <Routes>
        <Route path="/admin" element={<Admin />} />

        {/* Wrap multiple components in a fragment or div */}
        <Route
          path="/"
          element={
            <>
              <Home />
            </>
          }
        />
      </Routes>
    </div>
  );
};

export default App;
